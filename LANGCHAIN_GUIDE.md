# LangChain Integration Guide

## Current Architecture (LangChain-Ready)

All agents now use **LangChain chains** instead of raw LLM calls:

### Agent Structure

```python
class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client.llm  # LangChain Ollama instance
        self._build_chain()        # Build chain on init
    
    def _build_chain(self):
        """Build LangChain chain."""
        prompt = ChatPromptTemplate.from_messages([...])
        self.chain = prompt | self.llm | StrOutputParser()
    
    def process(self, input_data):
        """Use chain with .invoke()"""
        return self.chain.invoke({"key": input_data})
```

### Agents Converted to LangChain

1. **DiarizationAgent** - Speaker labeling chain
2. **UnifiedAnalysisAgent** - Intent/sentiment extraction chain  
3. **InsightAgent** - Business insight generation chain
4. **RootCauseAgent** - Statistical (no LLM, uses HDBSCAN)

### Chain Pattern

```
ChatPromptTemplate | LLM | OutputParser
```

**Benefits:**
- Composable chains
- Easy to add tools later
- Ready for LangGraph nodes
- Standardized error handling
- Prompt versioning

---

## Testing LangChain Integration

### Test 1: LLM Client
```bash
python test_langchain.py
```

### Test 2: Diarization Agent
```bash
python test_diarization.py
```

### Test 3: Full Pipeline
```bash
python test_pipeline.py
```

---

## Future: LangGraph Migration

When complexity increases (conditional routing, loops, human-in-loop):

### Step 1: Define State
```python
from langgraph.graph import StateGraph
from typing import TypedDict

class CallAnalysisState(TypedDict):
    audio_path: str
    transcript: str
    labeled_transcript: str
    analysis: dict
    insights: dict
    errors: list
```

### Step 2: Convert Agents to Nodes
```python
def diarization_node(state: CallAnalysisState):
    """LangGraph node wrapper."""
    agent = DiarizationAgent(llm_client)
    result = agent.label_speakers(state['transcript'])
    state['labeled_transcript'] = result['labeled_transcript']
    return state

def analysis_node(state: CallAnalysisState):
    agent = UnifiedAnalysisAgent(llm_client)
    state['analysis'] = agent.analyze(state['labeled_transcript'])
    return state
```

### Step 3: Build Graph
```python
workflow = StateGraph(CallAnalysisState)

# Add nodes
workflow.add_node("transcribe", transcribe_node)
workflow.add_node("diarize", diarization_node)
workflow.add_node("analyze", analysis_node)

# Add edges
workflow.add_edge("transcribe", "diarize")
workflow.add_edge("diarize", "analyze")

# Conditional routing (example)
def should_escalate(state):
    sentiment = state['analysis']['sentiment']['overall']
    return "escalate" if sentiment == "negative" else "complete"

workflow.add_conditional_edges(
    "analyze",
    should_escalate,
    {
        "escalate": "human_review",
        "complete": END
    }
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"audio_path": "call.wav"})
```

---

## When to Migrate to LangGraph

**Migrate when you need:**

1. **Conditional Routing**
   - If sentiment negative → escalate to human
   - If unresolved → retry with different agent

2. **Loops/Retry Logic**
   - Re-analyze if confidence low
   - Iterative refinement

3. **Human-in-the-Loop**
   - Pause for manual review
   - Approve/reject actions

4. **Parallel Processing**
   - Run sentiment + intent extraction in parallel
   - Multi-model consensus

5. **Complex Orchestration**
   - Multi-step workflows with branching
   - State management across steps

---

## Current vs Future

| Feature | Current (LangChain) | Future (LangGraph) |
|---------|---------------------|-------------------|
| Agent structure | ✅ Chains | ✅ Nodes |
| Linear flow | ✅ Orchestrator | ✅ Graph edges |
| Conditional logic | ❌ Python if/else | ✅ Conditional edges |
| Loops | ❌ Manual | ✅ Built-in |
| State management | ❌ Dict passing | ✅ StateGraph |
| Visualization | ❌ None | ✅ Graph view |
| Retry logic | ❌ Try/catch | ✅ Node retries |

---

## Migration Effort Estimate

**Current setup → LangGraph: 2-3 hours**

1. Define state schema (30 min)
2. Wrap agents as nodes (1 hour)
3. Build graph (30 min)
4. Add conditional edges (30 min)
5. Test (30 min)

**Already done:**
- ✅ Agents use LangChain chains
- ✅ Clean interfaces
- ✅ Error handling
- ✅ State is serializable

---

## Recommendation

**For MVP/Portfolio:**
- ✅ Current structure is perfect
- Shows you understand both approaches
- Interview talking point: "Built with LangChain chains, designed for easy LangGraph migration"

**For Production (>1000 calls/day):**
- Add LangGraph for:
  - Error recovery
  - Quality gates
  - Human review workflows
  - A/B testing different agents

---

## Files Modified

```
core/llm_client.py           → LangChain Ollama wrapper
agents/diarization_agent.py  → Chain-based speaker labeling
agents/unified_analysis_agent.py → Chain-based analysis
agents/insight_agent.py      → Chain-based insights
agents/root_cause_agent.py   → (No change, statistical)
core/orchestrator.py         → Uses LangChain agents
```

**Backward compatible:** All existing code works as before.
