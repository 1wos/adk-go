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

package console

import (
	"reflect"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestCollectPendingInterrupts_DetectionByLongRunningToolIDs
// verifies the detection contract: an event is a HITL prompt
// iff it has a non-empty LongRunningToolIDs and one of its
// content parts is a FunctionCall whose ID is in that set. The
// function name is not the discriminator — workflow input and
// any future kind all flow through the same detection path.
func TestCollectPendingInterrupts_DetectionByLongRunningToolIDs(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []pendingInterrupt
	}{
		{
			name:   "empty event list",
			events: nil,
			want:   nil,
		},
		{
			name: "event with FunctionCall but no LongRunningToolIDs is not an interrupt",
			events: []*session.Event{
				{
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "x", Name: "regular_tool"},
							}},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "event with LongRunningToolIDs but no matching FunctionCall is not an interrupt",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"abc"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{ID: "different_id", Name: "unmatched"},
							}},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "workflow input on Event.LLMResponse.Content is detected",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"int-1"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "int-1",
									Name: workflow.WorkflowInputFunctionCallName,
									Args: map[string]any{"message": "ok?"},
								},
							}},
						},
					},
				},
			},
			want: []pendingInterrupt{
				{id: "int-1", name: workflow.WorkflowInputFunctionCallName, args: map[string]any{"message": "ok?"}},
			},
		},
		{
			name: "multiple events, only ones with matching IDs surface",
			events: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "intro"}}}}},
				{
					LongRunningToolIDs: []string{"int-2"},
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "int-2", Name: "x"}}},
						},
					},
				},
				{LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "outro"}}}}},
			},
			want: []pendingInterrupt{
				{id: "int-2", name: "x", args: nil},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := collectPendingInterrupts(tc.events)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("collectPendingInterrupts() = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// TestBuildInterruptResponse_WorkflowInput verifies the
// workflow-input dispatch: a JSON object reply is returned
// verbatim (the operator submitted the final shape); scalars,
// arrays, and raw text are wrapped under "payload".
func TestBuildInterruptResponse_WorkflowInput(t *testing.T) {
	tests := []struct {
		name         string
		userInput    string
		wantResponse map[string]any
	}{
		{
			name:         "raw text is wrapped under payload",
			userInput:    "approve\n",
			wantResponse: map[string]any{"payload": "approve"},
		},
		{
			name:         "JSON object is returned verbatim (no wrapper)",
			userInput:    `{"approved": true, "days": 3}` + "\n",
			wantResponse: map[string]any{"approved": true, "days": float64(3)},
		},
		{
			name:         "JSON scalar is wrapped under payload",
			userInput:    "42\n",
			wantResponse: map[string]any{"payload": float64(42)},
		},
		{
			name:         "JSON array is wrapped under payload",
			userInput:    `[1, 2, "three"]` + "\n",
			wantResponse: map[string]any{"payload": []any{float64(1), float64(2), "three"}},
		},
		{
			name:         "trailing CR is stripped",
			userInput:    "approve\r\n",
			wantResponse: map[string]any{"payload": "approve"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := pendingInterrupt{id: "x", name: workflow.WorkflowInputFunctionCallName}
			part := buildInterruptResponse(p, tc.userInput)
			if part.FunctionResponse == nil {
				t.Fatalf("expected FunctionResponse part, got %+v", part)
			}
			if got, want := part.FunctionResponse.ID, "x"; got != want {
				t.Errorf("ID = %q, want %q", got, want)
			}
			if got, want := part.FunctionResponse.Name, workflow.WorkflowInputFunctionCallName; got != want {
				t.Errorf("Name = %q, want %q", got, want)
			}
			if !reflect.DeepEqual(part.FunctionResponse.Response, tc.wantResponse) {
				t.Errorf("Response = %#v, want %#v",
					part.FunctionResponse.Response, tc.wantResponse)
			}
		})
	}
}

// TestBuildInterruptResponse_GenericFallback verifies the catch-all
// path used for any long-running call name the launcher does not
// specifically know about.
func TestBuildInterruptResponse_GenericFallback(t *testing.T) {
	tests := []struct {
		name         string
		userInput    string
		wantResponse map[string]any
	}{
		{
			name:         "raw text is wrapped under result",
			userInput:    "some_value\n",
			wantResponse: map[string]any{"result": "some_value"},
		},
		{
			name:         "JSON object is returned verbatim (no wrapper)",
			userInput:    `{"foo": "bar"}` + "\n",
			wantResponse: map[string]any{"foo": "bar"},
		},
		{
			name:         "JSON scalar is wrapped under result",
			userInput:    "42\n",
			wantResponse: map[string]any{"result": float64(42)},
		},
		{
			name:         "JSON array is wrapped under result",
			userInput:    `[1, 2]` + "\n",
			wantResponse: map[string]any{"result": []any{float64(1), float64(2)}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := pendingInterrupt{id: "g", name: "some_unknown_kind"}
			part := buildInterruptResponse(p, tc.userInput)
			if !reflect.DeepEqual(part.FunctionResponse.Response, tc.wantResponse) {
				t.Errorf("Response = %#v, want %#v",
					part.FunctionResponse.Response, tc.wantResponse)
			}
		})
	}
}
