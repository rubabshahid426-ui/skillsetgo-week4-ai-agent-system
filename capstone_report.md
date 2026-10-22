# Capstone Report — AI Agent System

## Objective
Build a prototype integrating retrieval, tool use, and multi-step processing.

## Architecture
1. User submits a question.
2. Intent router chooses calculator or knowledge retrieval.
3. Calculator performs restricted arithmetic, or retrieval finds relevant knowledge-base passages.
4. Evidence/tool output is displayed.
5. User reviews the evidence before relying on the result.

## Test Plan
| Test | Input | Expected Result | Actual Result |
|---|---|---|---|
| Retrieval | What is RAG? | Relevant passage appears | Fill after running |
| Calculator | calculate 12 * (5 + 2) | 84 | Fill after running |
| Missing topic | Ask about absent knowledge | No matching evidence message | Fill after running |
| Empty input | Click run with blank input | Validation message | Fill after running |

## Limitations
This is a rule-based demo. Retrieval uses TF-IDF and the app does not generate answers with an LLM. Describe these limits accurately and do not claim unrun tests.
