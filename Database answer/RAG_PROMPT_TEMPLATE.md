# RAG Prompt Template

This document shows the exact prompt template used by the RAG system.

## Template Structure

```
You are a helpful assistant that answers questions based ONLY on the provided context.

CONTEXT:
{retrieved_chunk_1}

---

{retrieved_chunk_2}

---

{retrieved_chunk_3}

QUESTION: {user_question}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above.
- If the answer cannot be found in the context, respond with exactly: "Answer not found in the documents."
- Do not use any external knowledge or information not present in the context.
- Be concise and accurate.

ANSWER:
```

## Example

### Input Context
```
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, natural language processing, and speech recognition.
```

### User Question
```
What is the relationship between machine learning and deep learning?
```

### Generated Prompt
```
You are a helpful assistant that answers questions based ONLY on the provided context.

CONTEXT:
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

---

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, natural language processing, and speech recognition.

QUESTION: What is the relationship between machine learning and deep learning?

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above.
- If the answer cannot be found in the context, respond with exactly: "Answer not found in the documents."
- Do not use any external knowledge or information not present in the context.
- Be concise and accurate.

ANSWER:
```

### Expected LLM Response
```
Deep learning is a subset of machine learning. Machine learning is a broader field that includes various approaches, and deep learning is one specific approach within machine learning that uses neural networks with multiple layers.
```

## Key Features

1. **Strict Context-Only**: Explicitly instructs LLM to use ONLY provided context
2. **Fallback Response**: Clear instruction to return "Answer not found in the documents." if answer is not in context
3. **No External Knowledge**: Explicitly forbids using information not in context
4. **Clear Structure**: Separates context, question, and instructions clearly

## Implementation Location

The prompt is constructed in `backend/rag.py` in the `construct_prompt()` method.


