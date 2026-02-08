---
title: "Prompt Engineering"
css: styles.css
author: "Maria A"
description: " How to design effective prompts for large language models."
tags: ["deep learning", "prompt engineering", "research"]
---

# Prompt Engineering 

### The Core Problem with "Vanilla" Prompts

A pre-trained model has vast knowledge but lacks specific intent. A simple, vague prompt like:
> `"Explain quantum computing."`
...will yield a generic, one-size-fits-all answer. It might be too simple for an expert, too complex for a beginner, or lack the specific focus you need.

**Prompt engineering** solves this by adding context, constraints, and clarity to your instructions.

---


### Ways to improve model responses

*Good questions get good answers. Poorly worded questions get poor answers.
*
### Structure of the prompt

**Prompt** a strategic framework you use to communicate your intent to an LLM. 

By consciously combining **Instruction, Context, Input Data, and Output Indicators**—and using **examples** for complex tasks—you move from getting a generic, unpredictable response to a precise, reliable, and usable result.

<img src="../../shared_assets/visuals/images/prompt structure.avif" alt="Prompt Structure" width="400"/>

The core components of a prompt structure are:

- **Instruction (The Task)**
This is the most basic element: a direct command telling the model what to do.
*   **Examples:** `"Write...", "Summarize...", "Translate...", "Classify...", "Explain..."`

- **Context (The Background Information)**
This provides the model with necessary background, data, or specifics to ground its response. It sets the stage.
*   **Examples:** `"You are an expert marine biologist...", "Based on the following article: [pastes article text]...", "The user is a beginner gardener..."`

- **Input Data (The Content to Process)**
This is the specific text, data, or question you want the model to work on.
*   **Examples:** `"The text to summarize is: '...'", "Translate this sentence: '...'", "My question is: ..."`

- **Output Indicator (The Format Guide)**
This specifies how you want the answer formatted. This is crucial for getting usable results, especially for automated tasks.
*   **Examples:** `"Output in JSON format.", "Provide a bulleted list.", "Write a three-sentence summary.", "Return a table with columns X and Y."`

<img src="../../shared_assets/visuals/images/prompt structure2.png" alt="Prompt Structure" width="600"/>

---

### From Simple to Complex Structures

**Level 1: Basic Prompt (Just Instruction + Input)**
This is often sufficient for very simple tasks.
**Structure:** `[Instruction] + [Input Data]`
*   **Prompt:** `Translate the following English text to French: 'Hello, how are you?'`
    *   *Instruction:* `Translate... to French`
    *   *Input Data:* `'Hello, how are you?'`

**Level 2: Structured Prompt (The Standard Formula)**
This is the most common and effective structure for reliable results.
**Structure:** `[Context] + [Instruction] + [Input Data] + [Output Indicator]`
*   **Prompt:**
    > `You are a helpful assistant for a professional branding agency.` **(Context)**
    > `Generate five potential brand names for a new company that sells eco-friendly coffee beans.` **(Instruction)**
    > `The company's values are: sustainability, luxury, and ethical sourcing.` **(Input Data)**
    > `Present the names in a numbered list with a one-sentence explanation for each.` **(Output Indicator)**

**Level 3: Advanced Prompt (Few-Shot Learning)**
This is the most powerful structure for complex or precise tasks. You show the model examples of exactly what you want.
**Structure:** `[Context] + [Demonstrations] + [New Input]`
*   **Prompt:**
    > `Convert the following questions into search queries.` **(Instruction)**
    > \
    > `Input: What is the capital of the country with the largest population in the world?` \
    > `Output: China population capital` \
    > \
    > `Input: Who painted the Mona Lisa and in what year?` \
    > `Output: Mona Lisa painter year` \
    > \
    > `Input: What are the symptoms of a vitamin D deficiency?` **(New Input)** \
    > `Output:`


*(The model will infer the pattern from the demonstrations and output something like: `vitamin D deficiency symptoms`)*

### A Practical Example: Improving a Prompt

Let's see how structure transforms a weak prompt into a powerful one.

**Task:** Get information about photosynthesis.

*   **Unstructured (Weak) Prompt:**
    `"Photosynthesis"`
    *   *Problem:* Too vague. The model might generate a textbook chapter, a poem, or a list of facts with no focus.

*   **Basic Structured Prompt:**
    `"Explain photosynthesis in simple terms for a 10-year-old."`
    *   *Improvement:* Adds **Instruction** (`Explain`) and **Context** (`for a 10-year-old`).

*   **Well-Structured Prompt:**
    > `"Act as a friendly high school biology teacher.` **(Context)**
    > `Explain the process of photosynthesis in simple terms.` **(Instruction)**
    > `Focus on the role of sunlight, water, and carbon dioxide.` **(Input Data - specific focus)**
    > `Your explanation should be three paragraphs long and use an analogy to a kitchen recipe."` **(Output Indicator)**
    *   *Improvement:* Adds a **role**, specific focus points, and a clear format guide.

**Key Best Practices for Prompt Structure**

1.  **Be Specific and Clear:** Ambiguity is the enemy of good outputs. Prefer "list three causes" over "talk about causes."
2.  **Place Instructions at the Beginning or End:** Models pay strong attention to the start and finish of a prompt.
3.  **Use Delimiters:** Use `###`, `"""`, or `---` to separate different parts of your prompt (like instruction from context), especially when pasting large blocks of text. This helps the model parse your intent.
4.  **Iterate:** Your first prompt is a draft. If the output isn't right, refine your structure based on what you got. Ask it to be more concise, more detailed, or focus on a different aspect.
5.  **Test Variations:** Small changes in wording or order can have big effects. Experiment with different phrasings and structures to see what works best for your specific task.


---
### Prompting strategies

Prompting strategies are the high-level *approaches* or *tactics* you use within a well-structured prompt to achieve a specific type of outcome. They are the "how" of getting the model to think and reason in a certain way.

Key prompting strategies include:

### 1. Zero-Shot, One-Shot, and Few-Shot Prompting
This strategy defines how many examples you provide.

*   **Zero-Shot:** You give the model a task without any examples. You rely entirely on its pre-existing knowledge and reasoning capabilities.
    *   **When to use:** For simple, straightforward tasks where the instruction is clear enough on its own (e.g., "Translate this sentence to French:").
*   **One-Shot:** You provide a single example of the task.
    *   **When to use:** To quickly establish a simple pattern or format without spending tokens on multiple examples.
*   **Few-Shot:** You provide multiple examples (typically 2-5) of the task. This is one of the most powerful strategies for complex tasks.
    *   **When to use:** To teach the model a complex pattern, a specific style, or a nuanced classification that is difficult to describe with instructions alone. (The search query example from the previous answer is Few-Shot).

### 2. Chain-of-Thought (CoT) Prompting
This strategy forces the model to break down a complex problem into intermediate steps before stating a final answer.

*   **How it works:** You add phrases like `"Think step by step"`, `"Let's reason through this"`, or `"Show your working"` to the prompt. For even better results, provide few-shot examples that *show* the model how to reason step-by-step.
*   **Why it works:** It prevents the model from making intuitive but incorrect leaps and forces it to simulate a logical reasoning process. This is incredibly effective for math, logic, and complex planning tasks.
*   **Example:**
    *   **Without CoT:** `"If a zoo has 15 lions and 7 escape, but 3 are found, how many are left?"` *(Model might incorrectly guess 8)*
    *   **With CoT:** `"If a zoo has 15 lions and 7 escape, but 3 are found, how many are left? Let's think through this step by step."`
    *   **The Model's Output:** `"First, 15 lions - 7 escaped = 8 lions left in the zoo. Then, 3 are found and returned, so 8 + 3 = 11 lions. The answer is 11."`

### 3. Self-Consistency / Self-Critique
This strategy involves asking the model to evaluate or improve its own output.

*   **How it works:** You engage in a multi-turn dialogue where you ask the model to critique its previous answer, identify potential flaws, or generate multiple possibilities and then choose the best one.
*   **Why it works:** It separates the "generation" phase from the "evaluation" phase, leveraging the model's broad knowledge to spot its own errors.
*   **Example:**
    1.  **First Prompt:** `"Write a short blog intro about renewable energy."`
    2.  **Follow-Up Prompt:** `"Review the intro you just wrote. Is it engaging enough for a young audience? Rewrite it to be more concise and exciting."`

### 4. Tree-of-Thoughts / Step-Back Prompting
This advanced strategy involves asking the model to first abstract the core principles or concepts before tackling a detailed problem.

*   **How it works:** You prompt the model to take a "step back" from the specific problem to reason about the general domain or rules that govern it.
*   **Why it works:** It helps the model avoid getting lost in details and leverage its fundamental knowledge more effectively.
*   **Example:**
    *   **Specific Question:** `"How did the invention of the printing press influence the distribution of scientific knowledge in 16th-century Europe?"`
    *   **Step-Back Prompt:** `"First, what are the general principles of how communication technology impacts the spread of ideas? Second, apply those principles to the invention of the printing press."`

### 5. Persona / Role Prompting
This strategy involves instructing the model to adopt a specific identity, expertise, or perspective.

*   **How it works:** You prefix your prompt with `"Act as a [Role]..."`.
*   **Why it works:** It primes the model's vast training data to prioritize information, style, and tone associated with that specific role, leading to more specialized and relevant outputs.
*   **Example:**
    *   **Without Persona:** `"Explain blockchain."`
    *   **With Persona:** `"Act as a seasoned financial advisor explaining blockchain to a wealthy, risk-averse client who is curious but skeptical. Focus on its implications for asset security and long-term investing, not technical details."`

### 6. Retrieval Augmented Generation (RAG) Pattern
While often a system architecture, the prompting strategy involves providing the model with external knowledge it wasn't trained on.

*   **How it works:** You first retrieve relevant information from a database, document, or web search, and then you include that text in your prompt's context.
*   **Why it works:** It grounds the model's responses in factual, specific data, overcoming its limitations of having outdated or general knowledge and drastically reducing "hallucinations."
*   **Example:**
    `"""
    Based on the following company policy document:
    [Paste the entire policy text here]
    Answer the following question: How many vacation days does an employee accrue per year after 5 years of service?
    """`

 **Choosing the Right Strategy**

| Your Goal | Recommended Strategy |
| :--- | :--- |
| **Simple, factual tasks** | **Zero-Shot Prompting** |
| **Teaching a format or pattern** | **Few-Shot Prompting** |
| **Math, logic, complex reasoning** | **Chain-of-Thought (CoT)** |
| **Improving quality, reducing bias/errors** | **Self-Consistency / Critique** |
| **Highly specialized, tone-specific output** | **Persona / Role Prompting** |
| **Answering questions about specific, unseen data** | **Retrieval Augmented Generation (RAG)** |

These strategies are often combined. For example, you can use **Few-Shot Chain-of-Thought** prompting by providing examples that show step-by-step reasoning, or use a **Persona** with **Self-Critique** (`"Act as an editor and improve this text..."`). The best prompt engineers mix and match these tactics to achieve their desired result.


---

## Documenting and evaluating prompts

The *real* difficulty of prompt engineering isn’t just writing “one good prompt,” it’s an **iterative design process**. The same idea, phrased differently, or embedded in a different strategy (zero-shot vs few-shot, chain-of-thought vs direct), can completely change the model’s output. 

To make this systematic instead of chaotic, you need a **prompt tracking and evaluation loop**.

## 1. Treat Prompts Like Code (Version & Track Them)

* **Keep prompts in files/notebooks** instead of typing ad-hoc into a console.
* Assign **IDs or versions** to each variant. For example:

  * `summarizer_v1_zero_shot`
  * `summarizer_v2_few_shot`
  * `summarizer_v3_chain_of_thought`
* Use **Git/GitHub** for version control so you can track edits and revert if a newer wording degrades output.

---

## 2. Build a Prompt Registry

A simple table (CSV, Notion, or a database) with columns like:

| Prompt ID | Prompt Text                                                                 | Strategy   | Input Data | Output   | Evaluation Score | Notes                      |
| --------- | --------------------------------------------------------------------------- | ---------- | ---------- | -------- | ---------------- | -------------------------- |
| v1        | "Summarize this…"                                                           | Zero-shot  | Article A  | …output… | 2/5              | Too vague                  |
| v2        | "Summarize in 3 bullet points, focusing on causes, effects, and solutions…" | Structured | Article A  | …output… | 4/5              | Clearer, but missed nuance |
| v3        | Few-shot with 2 examples                                                    | Few-shot   | Article A  | …output… | 5/5              | Consistent and precise     |

This lets you **compare systematically** rather than by memory.

---

## 3. Define Evaluation Criteria

Before comparing, decide *what “good” means* for your use case. Examples:

* **Accuracy / factuality** (is it correct?)
* **Relevance** (does it answer the question, not drift off-topic?)
* **Structure** (does it follow the requested format?)
* **Tone/style match** (does it sound like your brand/voice?)
* **Consistency** (similar inputs → similar outputs)

You can score outputs manually (1–5 scale) or use automatic metrics if outputs are text summaries, translations, etc. (BLEU, ROUGE, BERTScore, etc.).

---

## 4. Automate Comparison

* For **quick experiments**, log prompt → output pairs in a spreadsheet.
* For **larger scale**, set up a **prompt evaluation pipeline**:

  * Run multiple prompts on the same test inputs.
  * Save outputs into structured JSON/CSV.
  * Auto-score them against metrics or feed them into another LLM as a “judge” (e.g., *“Rate these outputs on clarity 1–5”*).

---

## 5. Select & Lock the Winner

Once you’ve compared:

* Keep the **highest-scoring prompt variant** as your “production” version.
* Archive others, but don’t delete — they may become useful in a different context.
* Document *why* it was selected (to guide future edits).

---

Example: **Prompt Documentation Table**

| Prompt\_ID | Prompt\_Text                                                                                                       | Prompt\_Strategy | Input\_Data                    | Model      | Output                             | Evaluation\_Criteria             | Evaluation\_Score | Tester | Notes                                          | Date       |
| ---------- | ------------------------------------------------------------------------------------------------------------------ | ---------------- | ------------------------------ | ---------- | ---------------------------------- | -------------------------------- | ----------------- | ------ | ---------------------------------------------- | ---------- |
| v1         | "Summarize this article in one paragraph."                                                                         | Zero-shot        | Article on retail sales trends | Gemini Pro | Generic summary, missed details    | Clarity, Relevance               | 2/5               | Maria  | Too vague, lacks focus                         | 2025-09-17 |
| v2         | "Summarize the article in 3 bullet points: causes, effects, solutions."                                            | Structured       | Same article                   | Gemini Pro | Clear bullet points, but shallow   | Accuracy, Structure              | 4/5               | Maria  | Format is correct, but content is basic        | 2025-09-17 |
| v3         | "Here are two examples of summaries. Use the same style to summarize this article."                                | Few-shot         | Same article                   | Gemini Pro | Consistent, deeper analysis        | Accuracy, Relevance, Consistency | 5/5               | Maria  | Best so far, reusable                          | 2025-09-17 |
| v4         | "Act as a market analyst. Provide a 200-word executive summary with data-driven insights and future implications." | Role-based       | Same article                   | Gemini Pro | Structured executive-style summary | Depth, Tone, Professionalism     | 5/5               | Maria  | Very strong, suitable for client-facing output | 2025-09-17 |

---

*Next*: Interactive prompting with Gemini in Google AI Studio [Link](./gemini_interactive.md) 
----------
