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
variants = {"without-" + name: " || ".join(f"p.{other} != nil" for other in conditions if other != name)
            for name in conditions}
variants["require-signature"] = f"({predicate}) && len(p.ThoughtSignature) > 0"
variants["allow-standalone-signature"] = predicate + " || len(p.ThoughtSignature) > 0"
variants["allow-standalone-metadata"] = predicate + " || p.PartMetadata != nil"
replacements = {name: f"\t\tcase {expression}:\n\t\t\tconverted.Parts = append(converted.Parts, p)"
                for name, expression in variants.items()}
replacements["mutate-code"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ExecutableCode != nil { p.ExecutableCode.Code = "" }\n\t\t\tconverted.Parts =')
replacements["mutate-tool-args"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ToolCall != nil { clear(p.ToolCall.Args) }\n\t\t\tconverted.Parts =')
replacements["mutate-tool-result"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tif p.ToolResponse != nil { clear(p.ToolResponse.Response) }\n\t\t\tconverted.Parts =')
replacements["mutate-attached-signature"] = old.replace("\t\t\tconverted.Parts =", '\t\t\tp.ThoughtSignature = nil\n\t\t\tconverted.Parts =')
results = []
try:
    for name, replacement in replacements.items():
        source.write_text(original[:match.start()] + replacement + original[match.end():])
        command = ["go", "test", "-race", "-mod=readonly", "-count=1", "-json", "./internal/llminternal",
                   "-run", "TestConvertForeignEvent_RetainsPayloadParts|TestContentsRequestProcessor_ForeignThoughts|TestContentsRequestProcessor_ForeignThoughtCurrentTurn"]
        test = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (evidence / f"mutation-{name}.jsonl").write_text(test.stdout)
        (evidence / f"mutation-{name}.stderr").write_text(test.stderr)
        events = [json.loads(line) for line in test.stdout.splitlines() if line.startswith("{")]
        failures = [event["Test"] for event in events if event.get("Action") == "fail" and event.get("Test")]
        assert test.returncode == 1 and failures, (name, test.returncode, test.stderr)
        assert not any(event.get("Action") == "build-fail" for event in events), name
        results.append({"mutation": name, "failing_tests": failures})
        print(f"CAUGHT {name}: {len(failures)} failing test/subtest events", flush=True)
finally:
    source.write_text(original)
    (evidence / "mutations.json").write_text(json.dumps(results, indent=2))
assert len(results) == 14
