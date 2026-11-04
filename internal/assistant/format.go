package assistant

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func formatCorePayload(heading string, data json.RawMessage, asJSON bool) string {
	if asJSON {
		var pretty bytes.Buffer
		if err := json.Indent(&pretty, data, "", "  "); err != nil {
			return heading + "\n" + string(data) + "\n"
		}
		return heading + "\n" + pretty.String() + "\n"
	}
	var parsed any
	if err := json.Unmarshal(data, &parsed); err != nil {
		return heading + "\n" + string(data) + "\n"
	}
	var builder strings.Builder
	fmt.Fprintf(&builder, "%s\n", heading)
	writeReadableValue(&builder, parsed, 0)
	return builder.String()
}

func writeReadableValue(builder *strings.Builder, value any, indent int) {
	pad := strings.Repeat("  ", indent)
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			child := typed[key]
			switch child.(type) {
			case map[string]any, []any:
				fmt.Fprintf(builder, "%s%s:\n", pad, key)
				writeReadableValue(builder, child, indent+1)
			default:
				fmt.Fprintf(builder, "%s%s: %s\n", pad, key, compactJSON(child))
			}
		}
	case []any:
		for _, item := range typed {
			if object, ok := item.(map[string]any); ok {
				if name, ok := object["name"]; ok {
					stateStr := ""
					if s, ok := object["state"].(string); ok && s != "" {
						stateStr = s
					} else if s, ok := object["status"].(string); ok && s != "" {
						stateStr = s
					}
					if stateStr != "" {
						fmt.Fprintf(builder, "%s- %s (%s)\n", pad, compactJSON(name), stateStr)
					} else {
						fmt.Fprintf(builder, "%s- %s\n", pad, compactJSON(name))
					}
					continue
				}
				if names, ok := object["names"].([]any); ok {
					parts := make([]string, 0, len(names))
					for _, name := range names {
						parts = append(parts, strings.TrimPrefix(fmt.Sprint(name), "/"))
					}
					label := strings.Join(parts, ", ")
					stateStr := ""
					if s, ok := object["state"].(string); ok && s != "" {
						stateStr = s
					} else if s, ok := object["status"].(string); ok && s != "" {
						stateStr = s
					}
					if stateStr != "" {
						fmt.Fprintf(builder, "%s- %s (%s)\n", pad, label, stateStr)
					} else {
						fmt.Fprintf(builder, "%s- %s\n", pad, label)
					}
					continue
				}
			}
			fmt.Fprintf(builder, "%s- %s\n", pad, compactJSON(item))
		}
	default:
		fmt.Fprintf(builder, "%s%s\n", pad, compactJSON(value))
	}
}

func compactJSON(value any) string {
	switch typed := value.(type) {
	case nil:
		return "null"
	case string:
		return typed
	case float64:
		if typed == float64(int64(typed)) {
			return fmt.Sprintf("%d", int64(typed))
		}
		return fmt.Sprintf("%g", typed)
	case bool:
		return fmt.Sprintf("%t", typed)
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return fmt.Sprint(value)
		}
		return string(encoded)
	}
}

func mutationEndpoint(spec mutationSpec, name string) string {
	switch spec.targetType {
	case "stack":
		if spec.action == "restart" {
			return fmt.Sprintf("POST /api/v2/stacks/%s/down then POST /api/v2/stacks/%s/up", name, name)
		}
		action := "up"
		if spec.action == "stop" {
			action = "down"
		}
		return fmt.Sprintf("POST /api/v2/stacks/%s/%s", name, action)
	case "service":
		return fmt.Sprintf("POST /api/v2/system/services/%s/%s", name, spec.action)
	default:
		return fmt.Sprintf("POST /api/v2/containers/%s/%s", name, spec.action)
	}
}
