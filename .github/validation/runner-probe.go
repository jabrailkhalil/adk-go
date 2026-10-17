// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This standalone, offline probe exercises real Runner/session/agent processing.
// Both model backends are local stubs; no live model or API is contacted.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"os"
	"reflect"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

type captureModel struct {
	responses []*genai.Content
	requests  []*model.LLMRequest
}

func (*captureModel) Name() string { return "offline-capture" }

func (m *captureModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		if len(m.requests) >= len(m.responses) {
			yield(nil, errors.New("unexpected model call"))
			return
		}
		// Snapshot the actual model-bound request before returning a response.
		// No request contents are printed or written to disk.
		request, err := clone(req)
		if err != nil {
			yield(nil, err)
			return
		}
		response, err := clone(m.responses[len(m.requests)])
		if err != nil {
			yield(nil, err)
			return
		}
		m.requests = append(m.requests, request)
		yield(&model.LLMResponse{Content: response}, nil)
	}
}

func clone[T any](v *T) (*T, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, errors.New("snapshot encode failed")
	}
	var result T
	if err := json.Unmarshal(b, &result); err != nil {
		return nil, errors.New("snapshot decode failed")
	}
	return &result, nil
}

type scenario struct {
	name               string
	foreign            []*genai.Part
	want               *genai.Content
	seedOwn            bool
	includeNone        bool
	finishAfterThought bool
}

func content(role string, parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: role, Parts: parts}
}

func runScenario(tc scenario) (err error) {
	// A regression in the caller's nil guard must fail this case without
	// dumping a panic or potentially sensitive event contents to the transcript.
	defer func() {
		if recover() != nil {
			err = errors.New("runner panicked")
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	const app, user, sessionID = "thought-probe", "probe-user", "probe-session"
	store := session.InMemoryService()
	created, err := store.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: sessionID})
	if err != nil {
		return errors.New("session create failed")
	}
	before := make(map[string]*session.Event)
	var wantContents []*genai.Content
	if tc.seedOwn {
		own := content("model", &genai.Part{Thought: true, Text: "own private fixture"}, &genai.Part{Text: "own public fixture"})
		ev := session.NewEvent(ctx, "previous-invocation")
		ev.Author = "reviewer"
		ev.LLMResponse.Content = own
		if err := store.AppendEvent(ctx, created.Session, ev); err != nil {
			return errors.New("history setup failed")
		}
		snapshot, err := clone(ev)
		if err != nil {
			return err
		}
		before[ev.ID] = snapshot
		wantContents = append(wantContents, own)
	}
	input := content("user", &genai.Part{Text: "review the previous answer"})
	wantContents = append(wantContents, input)
	if tc.want != nil {
		wantContents = append(wantContents, tc.want)
	}
	producer := &captureModel{responses: []*genai.Content{content("model", tc.foreign...)}}
	if tc.finishAfterThought {
		// Flow asks for another turn after a thought-only response. Finish it
		// normally so this tests filtering rather than the retry-limit guard.
		producer.responses = append(producer.responses, content("model", &genai.Part{Text: "producer complete"}))
		wantContents = append(wantContents, content("user", &genai.Part{Text: "For context:"}, &genai.Part{Text: "[first] said: producer complete"}))
	}
	reviewer := &captureModel{responses: []*genai.Content{content("model", &genai.Part{Text: "review complete"})}}
	first, err := llmagent.New(llmagent.Config{Name: "first", Model: producer, IncludeContents: llmagent.IncludeContentsDefault})
	if err != nil {
		return errors.New("producer setup failed")
	}
	reviewerConfig := llmagent.Config{Name: "reviewer", Model: reviewer, IncludeContents: llmagent.IncludeContentsDefault}
	if tc.includeNone {
		reviewerConfig.IncludeContents = "none"
	}
	second, err := llmagent.New(reviewerConfig)
	if err != nil {
		return errors.New("reviewer setup failed")
	}
	sequence, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{first, second}}})
	if err != nil {
		return errors.New("sequence setup failed")
	}
	r, err := runner.New(runner.Config{AppName: app, Agent: sequence, SessionService: store})
	if err != nil {
		return errors.New("runner setup failed")
	}
	emitted, producerEvents := 0, 0
	for ev, runErr := range r.Run(ctx, user, sessionID, input, agent.RunConfig{}) {
		if runErr != nil || ev == nil {
			return errors.New("runner failed")
		}
		emitted++
		if ev.Author == "first" {
			if producerEvents >= len(producer.responses) || !reflect.DeepEqual(ev.Content, producer.responses[producerEvents]) {
				return errors.New("producer event contents mismatch")
			}
			producerEvents++
			snapshot, err := clone(ev)
			if err != nil {
				return err
			}
			before[ev.ID] = snapshot
		}
	}
	if len(producer.requests) != len(producer.responses) || len(reviewer.requests) != 1 || emitted != len(producer.responses)+1 {
		return errors.New("unexpected model or event count")
	}
	if !reflect.DeepEqual(reviewer.requests[0].Contents, wantContents) {
		return errors.New("outbound contents mismatch")
	}
	stored, err := store.Get(ctx, &session.GetRequest{AppName: app, UserID: user, SessionID: sessionID})
	if err != nil {
		return errors.New("session read failed")
	}
	wantEvents, wantSnapshots := len(producer.responses)+2, len(producer.responses)
	if tc.seedOwn {
		wantEvents++
		wantSnapshots++
	}
	if stored.Session.Events().Len() != wantEvents || len(before) != wantSnapshots {
		return errors.New("stored event count mismatch")
	}
	matched := 0
	for ev := range stored.Session.Events().All() {
		if prior, ok := before[ev.ID]; ok {
			snapshot, err := clone(ev)
			if err != nil {
				return err
			}
			if !reflect.DeepEqual(snapshot, prior) {
				return errors.New("stored event mutated")
			}
			matched++
		}
	}
	if matched != wantSnapshots {
		return errors.New("stored event missing")
	}
	return nil
}

func main() {
	relayed := content("user", &genai.Part{Text: "For context:"}, &genai.Part{Text: "[first] said: public fixture"})
	image := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2}}, ThoughtSignature: []byte{3, 4}}
	cases := []scenario{
		{name: "empty_foreign_part_does_not_cut_out_user", foreign: []*genai.Part{{}}, includeNone: true},
		{name: "foreign_signature_does_not_cut_out_user", foreign: []*genai.Part{{ThoughtSignature: []byte{1, 2}}}, includeNone: true},
		{name: "foreign_metadata_does_not_cut_out_user", foreign: []*genai.Part{{PartMetadata: map[string]any{"source": "test"}}}, includeNone: true},
		{name: "public_image_keeps_attached_signature", foreign: []*genai.Part{{}, {ThoughtSignature: []byte{1, 2}}, image}, want: content("user", &genai.Part{Text: "For context:"}, image)},
		{name: "ordinary_foreign_text_retained", foreign: []*genai.Part{{Text: "public fixture"}}, want: relayed},
		{name: "own_agent_thought_unchanged", foreign: []*genai.Part{{Text: "public fixture"}}, want: relayed, seedOwn: true},
	}
	passed := 0
	for _, tc := range cases {
		if err := runScenario(tc); err != nil {
			fmt.Printf("FAIL %s: %v\n", tc.name, err)
			continue
		}
		fmt.Printf("PASS %s\n", tc.name)
		passed++
	}
	fmt.Printf("PASS %d FAIL %d TOTAL %d\n", passed, len(cases)-passed, len(cases))
	if passed != len(cases) {
		os.Exit(1)
	}
}
