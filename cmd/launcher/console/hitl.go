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
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/adk/workflow"
)

// pendingInterrupt is one HITL prompt the agent emitted on the
// previous turn that still needs a user reply on the next. The
// id+name pair come from a FunctionCall part whose ID appears
// in Event.LongRunningToolIDs.
type pendingInterrupt struct {
	id   string
	name string
	args map[string]any
}

// collectPendingInterrupts scans events for FunctionCall parts
// referenced from LongRunningToolIDs on the same event and
// returns them in order of appearance.
func collectPendingInterrupts(events []*session.Event) []pendingInterrupt {
	var out []pendingInterrupt
	for _, ev := range events {
		if ev == nil || len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		lr := map[string]struct{}{}
		for _, id := range ev.LongRunningToolIDs {
			lr[id] = struct{}{}
		}
		c := ev.Content
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			fc := p.FunctionCall
			if fc == nil {
				continue
			}
			if _, isInterrupt := lr[fc.ID]; !isInterrupt {
				continue
			}
			out = append(out, pendingInterrupt{
				id:   fc.ID,
				name: fc.Name,
				args: fc.Args,
			})
		}
	}
	return out
}

// renderInterruptPrompt prints the pending interrupt and a
// "User -> " input prompt to stdout. The caller reads the user's
// reply and passes it to buildInterruptResponse.
func renderInterruptPrompt(p pendingInterrupt) {
	switch p.name {
	case workflow.WorkflowInputFunctionCallName:
		renderWorkflowInputPrompt(p.args)
	case toolconfirmation.FunctionCallName:
		renderToolConfirmationPrompt(p.args)
	default:
		renderGenericInterruptPrompt(p.name, p.args)
	}
	fmt.Print("User -> ")
}

// buildInterruptResponse converts the operator's one-line input
// into a FunctionResponse part keyed by the interrupt's id and
// name. Same per-name dispatch as renderInterruptPrompt.
func buildInterruptResponse(p pendingInterrupt, userInput string) *genai.Part {
	line := strings.TrimRight(userInput, "\r\n")

	var response map[string]any
	switch p.name {
	case workflow.WorkflowInputFunctionCallName:
		response = workflowInputResponseFromUserInput(line)
	case toolconfirmation.FunctionCallName:
		response = toolConfirmationResponseFromUserInput(line)
	default:
		response = genericResponseFromUserInput(line)
	}
	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       p.id,
			Name:     p.name,
			Response: response,
		},
	}
}

// renderWorkflowInputPrompt prints the workflow request prompt.
// Args carry the fields populated by workflow.NewRequestInputEvent:
// interruptId, message, payload, responseSchema.
func renderWorkflowInputPrompt(args map[string]any) {
	msg, _ := args["message"].(string)
	if msg == "" {
		msg = "Input requested"
	}
	fmt.Printf("Agent -> %s\n", msg)
	if payload, ok := args["payload"]; ok && payload != nil {
		if pretty, err := json.MarshalIndent(payload, "  ", "  "); err == nil {
			fmt.Printf("  Payload: %s\n", pretty)
		} else {
			fmt.Printf("  Payload: %v\n", payload)
		}
	}
	if schema, ok := args["responseSchema"]; ok && schema != nil {
		if pretty, err := json.Marshal(schema); err == nil {
			fmt.Printf("  Expected response schema: %s\n", pretty)
		}
	}
}

// workflowInputResponseFromUserInput shapes a one-line operator
// reply into a workflow-input FunctionResponse payload. Tries
// JSON first; a parsed object is returned verbatim so the
// operator can submit a fully-structured response, scalars and
// arrays are wrapped under "payload", and unparseable input is
// passed through under "payload" too.
func workflowInputResponseFromUserInput(line string) map[string]any {
	var parsed any
	if err := json.Unmarshal([]byte(line), &parsed); err == nil {
		if asMap, ok := parsed.(map[string]any); ok {
			return asMap
		}
		return map[string]any{"payload": parsed}
	}
	return map[string]any{"payload": line}
}

// renderToolConfirmationPrompt prints the tool-confirmation
// prompt. Recognised arg keys: "toolConfirmation" (a map with
// a "hint" string) and "originalFunctionCall" (a map with a
// "name" string used as fallback prompt).
func renderToolConfirmationPrompt(args map[string]any) {
	hint := ""
	if tc, ok := args["toolConfirmation"].(map[string]any); ok {
		hint, _ = tc["hint"].(string)
	}
	if hint == "" {
		originalName := "unknown"
		if oc, ok := args["originalFunctionCall"].(map[string]any); ok {
			if name, _ := oc["name"].(string); name != "" {
				originalName = name
			}
		}
		hint = "Confirm " + originalName + "?"
	}
	fmt.Printf("Agent -> %s\n", hint)
	fmt.Println("  Type 'yes' to confirm, anything else to reject.")
}

// toolConfirmationResponseFromUserInput maps yes-ish answers
// (y/yes/true/confirm, case-insensitive) to {"confirmed": true};
// everything else (including blank lines) maps to
// {"confirmed": false}.
func toolConfirmationResponseFromUserInput(line string) map[string]any {
	answer := strings.TrimSpace(strings.ToLower(line))
	switch answer {
	case "y", "yes", "true", "confirm":
		return map[string]any{"confirmed": true}
	default:
		return map[string]any{"confirmed": false}
	}
}

// renderGenericInterruptPrompt is the fallback for HITL kinds the
// launcher does not specifically recognise. Prints the kind name
// and the raw args so the operator can compose a sensible
// response by hand.
func renderGenericInterruptPrompt(name string, args map[string]any) {
	fmt.Printf("Agent -> waiting for response (kind: %s)\n", name)
	if len(args) > 0 {
		if pretty, err := json.Marshal(args); err == nil {
			fmt.Printf("  Args: %s\n", pretty)
		} else {
			fmt.Printf("  Args: %v\n", args)
		}
	}
}

// genericResponseFromUserInput shapes the operator's reply for
// HITL kinds without a dedicated parser. A parsed JSON object is
// returned verbatim so the operator can submit a fully-structured
// response; scalars and arrays are wrapped under "result"; raw
// text falls back to "result" too.
func genericResponseFromUserInput(line string) map[string]any {
	var parsed any
	if err := json.Unmarshal([]byte(line), &parsed); err == nil {
		if asMap, ok := parsed.(map[string]any); ok {
			return asMap
		}
		return map[string]any{"result": parsed}
	}
	return map[string]any{"result": line}
}
