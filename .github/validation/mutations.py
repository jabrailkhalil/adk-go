import json
import re
import subprocess
from pathlib import Path

source = Path("internal/llminternal/contents_processor.go")
original = source.read_text()
evidence = Path("../evidence")
match = re.search(r"\t\tcase p\.InlineData != nil.*?\n\t\t\tconverted\.Parts = append\(converted\.Parts, p\)", original, re.S)
assert match
old = match.group()
conditions = ["InlineData", "FileData", "ExecutableCode", "CodeExecutionResult", "ToolCall", "ToolResponse", "AudioTranscription"]
predicate = " || ".join(f"p.{name} != nil" for name in conditions)
metadata_fields = {
    "PartMetadata": "p.PartMetadata != nil",
    "VideoMetadata": "p.VideoMetadata != nil",
    "MediaResolution": "p.MediaResolution != nil",
    "MediaProcessing": 'p.MediaProcessing != ""',
    "SpeechMetadata": "p.SpeechMetadata != nil",
}
has_payload = '''\thasPayload := false
\tfor _, part := range content.Parts {
\t\tif part != nil && !part.Thought && (%s) {
\t\t\thasPayload = true
\t\t\tbreak
\t\t}
\t}

''' % ('part.Text != "" || part.FunctionCall != nil || part.FunctionResponse != nil || ' + predicate.replace("p.", "part."))
variants = {"without-" + name: " || ".join(f"p.{other} != nil" for other in conditions if other != name)
            for name in conditions}
variants["require-signature"] = f"({predicate}) && len(p.ThoughtSignature) > 0"
variants["allow-standalone-signature"] = predicate + " || len(p.ThoughtSignature) > 0"
variants["allow-standalone-metadata"] = predicate + " || p.PartMetadata != nil"
for name in ["VideoMetadata", "MediaResolution", "SpeechMetadata"]:
    variants["allow-standalone-" + name] = predicate + f" || p.{name} != nil"
variants["allow-standalone-MediaProcessing"] = predicate + ' || p.MediaProcessing != ""'
required_metadata = " || ".join(expression for name, expression in metadata_fields.items() if name != "PartMetadata")
variants["require-attached-metadata"] = f"({predicate}) && ({required_metadata})"
mixed_names = {"allow-mixed-" + name for name in metadata_fields}
for name, expression in metadata_fields.items():
    variants["allow-mixed-" + name] = f"{predicate} || (hasPayload && {expression})"
replacements = {name: f"\t\tcase {expression}:\n\t\t\tconverted.Parts = append(converted.Parts, p)"
                for name, expression in variants.items()}
replacements["mutate-code"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ExecutableCode != nil { p.ExecutableCode.Code = "" }\n\t\t\tconverted.Parts =')
replacements["mutate-tool-args"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ToolCall != nil { clear(p.ToolCall.Args) }\n\t\t\tconverted.Parts =')
replacements["mutate-tool-result"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ToolResponse != nil { clear(p.ToolResponse.Response) }\n\t\t\tconverted.Parts =')
replacements["mutate-attached-signature"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tp.ThoughtSignature = nil\n\t\t\tconverted.Parts =')
for name in ["VideoMetadata", "MediaResolution", "SpeechMetadata"]:
    replacements["drop-attached-" + name] = old.replace(
        "\t\t\tconverted.Parts =", f"\t\t\tp.{name} = nil\n\t\t\tconverted.Parts =")
replacements["drop-attached-MediaProcessing"] = old.replace(
    "\t\t\tconverted.Parts =", '\t\t\tp.MediaProcessing = ""\n\t\t\tconverted.Parts =')
results = []
try:
    for name, replacement in replacements.items():
        mutated = original[:match.start()] + replacement + original[match.end():]
        if name in mixed_names:
            position = mutated.index("\tconverted := &genai.Content{", mutated.index("func ConvertForeignEvent("))
            mutated = mutated[:position] + has_payload + mutated[position:]
        source.write_text(mutated)
        command = ["go", "test", "-race", "-mod=readonly", "-count=1", "-json", "./internal/llminternal",
                   "-run", "TestConvertForeignEvent_RetainsPayloadParts|TestContentsRequestProcessor_ForeignThoughts|TestContentsRequestProcessor_ForeignThoughtCurrentTurn"]
        test = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (evidence / f"mutation-{name}.jsonl").write_text(test.stdout)
        (evidence / f"mutation-{name}.stderr").write_text(test.stderr)
        events = [json.loads(line) for line in test.stdout.splitlines() if line.startswith("{")]
        failures = [event["Test"] for event in events if event.get("Action") == "fail" and event.get("Test")]
        assert test.returncode == 1 and failures, (name, test.returncode, test.stderr)
        assert not any(event.get("Action") == "build-fail" for event in events), name
        leaves = [event["Test"] for event in events if event.get("Action") == "fail" and "/" in event.get("Test", "")]
        retention = [leaf for leaf in leaves if leaf.startswith("TestConvertForeignEvent_RetainsPayloadParts/")]
        if name == "require-attached-metadata":
            assert len(retention) == 14 and all("with_metadata" not in leaf for leaf in retention), leaves
            assert len(leaves) == 15 and "TestContentsRequestProcessor_ForeignThoughts/mixed_foreign_event_retains_public_parts" in leaves, leaves
        elif name in mixed_names:
            assert len(retention) == 28 and len(leaves) == 28, leaves
        results.append({"mutation": name, "failing_tests": failures})
        print(f"CAUGHT {name}: {len(failures)} failing test/subtest events", flush=True)
finally:
    source.write_text(original)
    (evidence / "mutations.json").write_text(json.dumps(results, indent=2))
assert len(results) == 28
