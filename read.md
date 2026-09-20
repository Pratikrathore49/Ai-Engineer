Bilkul Pratik. **Fine-tuning** Generative AI ka bahut important topic hai, aur interview mein aksar poocha jata hai. Main ise pehle simple life example se samjhaunga, phir technical level par, aur end mein **perfect interview answer** dunga.

# 1. Fine-tuning kya hota hai?

Simple definition:

> **Fine-tuning means taking an already pretrained AI model and training it further on a smaller, task-specific dataset so that it behaves better for a particular task, style, or domain.**

Matlab hum model ko zero se train nahi karte.

Humare paas already ek trained model hai, jaise:

```text
Llama
Mistral
Qwen
GPT-type models
```

Ye model pehle se language samajhna, text generate karna, patterns identify karna seekh chuka hai.

Fine-tuning mein hum usko bolte hain:

> "Tumhe general language aati hai, ab is particular type ke kaam ko aur achhe se seekho."

---

# 2. General life example

Maan lo ek student hai.

Usne school mein:

* English seekhi
* Maths seekhi
* Science seekhi
* General knowledge seekha

Ab us student ko **CA exam** dena hai.

Usko dobara alphabet se English nahi sikhani.

Instead:

```text
Already learned:
General education

        ↓

Specialized training:
CA-specific questions
CA syllabus
CA exam pattern
CA answer style

        ↓

Result:
Better CA performance
```

AI model ke saath bhi same concept hai.

```text
Pretrained Model
      ↓
General language + knowledge + patterns
      ↓
Fine-tuning dataset
      ↓
Specialized behavior
```

---

# 3. Pretraining vs Fine-tuning

Ye difference interview ke liye extremely important hai.

### Pretraining

Model ko **bahut huge dataset** par train kiya jata hai.

Example:

```text
Books
Web pages
Articles
Code
Documents
Conversations
etc.
```

Goal:

> Model ko general language patterns aur knowledge seekhana.

Simplified:

```text
Huge Data
   ↓
Pretraining
   ↓
Base Model
```

### Fine-tuning

Ab base model ko **smaller specialized dataset** par further train karte hain.

```text
Base Model
   +
Task-specific Dataset
   ↓
Fine-tuned Model
```

Example:

```text
Base LLM
   ↓
10,000 customer-support conversations
   ↓
Customer-support fine-tuned LLM
```

---

# 4. Fine-tuning actually model ke andar karta kya hai?

Ye sabse important technical point hai.

LLM ke andar millions/billions of **parameters** hote hain.

Example:

```text
Model parameters
w1
w2
w3
...
wN
```

Training ke time model prediction karta hai.

Suppose input:

```text
User: I want to cancel my order.
```

Expected response:

```text
Assistant: Sure, I can help you cancel your order.
```

Model prediction aur correct answer ke beech difference calculate hota hai.

Is difference ko **loss** bolte hain.

```text
Input
 ↓
Model
 ↓
Prediction
 ↓
Compare with correct answer
 ↓
Loss
 ↓
Backpropagation
 ↓
Parameter updates
```

Fine-tuning ke dauraan parameters adjust hote hain so that model future mein similar examples par better response de.

---

# 5. Ek simple example

Suppose pretrained model normally answer karta hai:

**User:**

```text
How can I return a product?
```

Model:

```text
You should contact the seller and check the return policy...
```

Ab company chahti hai ki uska support assistant specific format mein answer kare.

Training examples:

```text
User:
How can I return a product?

Assistant:
To return your product:

1. Open My Orders
2. Select the product
3. Click Return
4. Choose a return reason
```

Aise thousands examples dene ke baad model gradually us specific response pattern ko learn kar sakta hai.

---

# 6. Fine-tuning ka matlab model ko new knowledge dena hai?

**Not exactly.**

Ye ek common misunderstanding hai.

Fine-tuning primarily useful hai:

* behavior
* style
* format
* task performance
* domain-specific patterns
* instruction following

ke liye.

For example:

```text
"Answer in JSON"

"Use a legal-document style"

"Generate SQL from natural language"

"Classify customer complaints"

"Follow our company's response format"
```

Ye fine-tuning ke strong use cases hain.

Lekin agar tumhare paas:

```text
Company ke 2 lakh private documents
```

hain aur tum chahte ho model unke latest facts ke according answer kare, to **RAG often better approach** hoti hai.

---

# 7. Fine-tuning vs RAG

Tumhare GenAI journey ke liye ye distinction bahut important hai.

### RAG

RAG mein model ko permanently train nahi karte.

Instead:

```text
User Question
      ↓
Retrieve relevant documents
      ↓
Give documents to LLM
      ↓
Generate answer
```

Example:

Company policy:

```text
Latest leave policy = 24 annual leaves
```

Agar policy change ho gayi:

```text
24 → 30
```

RAG system mein database/document update kar sakte ho.

### Fine-tuning

Fine-tuning mein model ko examples se **behavior/task pattern** sikhate ho.

For example:

```text
Input → Desired Output

10,000 examples
        ↓
Fine-tune
        ↓
Model learns the pattern
```

### Easy memory trick

> **RAG gives the model information at inference time.**
> **Fine-tuning changes the model's learned behavior/parameters.**

---

# 8. Example: company chatbot

Suppose company has:

```text
10,000 HR documents
```

and wants chatbot.

### Question: "What is our leave policy?"

Use:

```text
RAG
```

because the chatbot needs company documents.

But suppose company wants:

> Every answer should be concise, professional, and always follow a specific JSON format.

This may be a good candidate for:

```text
Fine-tuning
```

Often real systems combine both:

```text
                ┌─────────────┐
User Question → │    RAG      │
                └──────┬──────┘
                       ↓
                  Relevant Data
                       ↓
                Fine-tuned LLM
                       ↓
                    Answer
```

So **RAG and fine-tuning are not competitors in every situation**. They can work together.

---

# 9. Fine-tuning dataset kaisa hota hai?

Usually training data input-output examples ke form mein hota hai.

For instruction tuning:

```json
{
  "instruction": "Summarize this text",
  "input": "Artificial intelligence is...",
  "output": "AI is..."
}
```

For chat models:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is Python?"
    },
    {
      "role": "assistant",
      "content": "Python is a programming language..."
    }
  ]
}
```

Thousands of high-quality examples model ko desired pattern sikhate hain.

---

# 10. High-quality data important kyun hai?

Ye rule yaad rakho:

> **Garbage in, garbage out.**

Suppose training data:

```text
5,000 bad examples
```

Then model undesirable behavior learn kar sakta hai.

Suppose:

```text
10,000 carefully written examples
```

Then fine-tuning much more useful ho sakti hai.

Dataset mein ideally hona chahiye:

```text
Correct answers
Consistent format
Relevant examples
Diverse cases
Clean data
```

---

# 11. Fine-tuning ke major types

Fine-tuning ek single technique nahi hai.

### 1. Supervised Fine-Tuning — SFT

Most common concept.

Hum model ko:

```text
Input → Correct Output
```

examples dete hain.

Example:

```text
Question:
What is RAG?

Answer:
RAG stands for Retrieval-Augmented Generation...
```

Model desired responses learn karta hai.

---

### 2. Parameter-Efficient Fine-Tuning — PEFT

Full model ke saare parameters update karne ke bajay, hum small number of additional/trainable parameters use karte hain.

Popular techniques:

```text
LoRA
QLoRA
Adapters
```

Ye modern LLM fine-tuning mein extremely important concepts hain.

---

# 12. LoRA kya hai?

Ye interview mein aa sakta hai.

**LoRA = Low-Rank Adaptation**

Idea:

Instead of changing billions of original parameters directly, hum model ke saath small trainable matrices add karte hain.

Simplified:

```text
Original Model
     +
Small trainable LoRA parameters
     ↓
Fine-tuned behavior
```

Advantages:

```text
Less GPU memory
Less training cost
Faster training
Smaller fine-tuned weights
```

Isliye practical LLM fine-tuning mein LoRA popular hai.

---

# 13. QLoRA kya hai?

QLoRA = roughly:

```text
Quantization + LoRA
```

Model ko lower precision mein load karke memory reduce karte hain aur LoRA adapters train karte hain.

Simplified:

```text
Large Model
   ↓
Quantized model
   +
LoRA adapters
   ↓
Fine-tuning
```

Iska major benefit:

> Large models ko comparatively limited GPU memory par fine-tune karna easier ho jata hai.

---

# 14. Full fine-tuning vs LoRA

### Full fine-tuning

```text
All/most model parameters
          ↓
Update
```

Advantages:

* Maximum flexibility

Disadvantages:

* Very expensive
* High GPU memory
* Large training requirements

### LoRA

```text
Original parameters mostly frozen
          +
Small trainable adapters
```

Advantages:

* Cheaper
* Faster
* Less memory
* Easier experimentation

---

# 15. Fine-tuning process

Production-level simplified workflow:

```text
1. Define task
      ↓
2. Collect data
      ↓
3. Clean data
      ↓
4. Format dataset
      ↓
5. Split train/validation data
      ↓
6. Choose base model
      ↓
7. Choose fine-tuning method
      ↓
8. Train
      ↓
9. Evaluate
      ↓
10. Deploy
      ↓
11. Monitor
```

Example:

```text
Base Model
   ↓
Customer support dataset
   ↓
SFT + LoRA
   ↓
Evaluation
   ↓
Deploy
```

---

# 16. Epoch kya hota hai?

Fine-tuning mein ye bhi common term hai.

Suppose dataset mein:

```text
10,000 examples
```

Model poore 10,000 examples ek baar process karta hai:

```text
1 epoch
```

Do baar:

```text
2 epochs
```

Three times:

```text
3 epochs
```

But more training ≠ always better.

Agar bahut zyada train kar diya:

```text
Overfitting
```

ho sakta hai.

---

# 17. Overfitting kya hai?

School example:

Ek student 100 questions ko itna yaad kar leta hai ki wahi exact questions aaye to perfect answer de.

Lekin question thoda change karo:

```text
Student struggles
```

This is similar to overfitting.

Model training examples ko memorize karne lagta hai instead of generalizing.

So:

```text
Too little training → underfitting
Too much training → overfitting
Balanced training → better generalization
```

---

# 18. Fine-tuning mein kya hyperparameters hote hain?

Common ones:

```text
Learning rate
Batch size
Number of epochs
Warmup steps
Weight decay
Gradient accumulation
```

For interview, especially know:

### Learning rate

Determines:

> Parameters ko kitna aggressively update karna hai.

Very high:

```text
Training unstable ho sakti hai
```

Very low:

```text
Training very slow ho sakti hai
```

---

# 19. Fine-tuning ka real-world example

Maan lo ek hospital-like organization chahti hai ki LLM:

> Medical reports ko predefined format mein summarize kare.

Training dataset:

```text
Report 1 → Desired summary
Report 2 → Desired summary
Report 3 → Desired summary
...
Report 20,000 → Desired summary
```

Fine-tuning ke baad model better learn kar sakta hai:

```text
Patient Info
Symptoms
Findings
Impression
Recommended follow-up
```

format.

But factual/current medical information ke liye retrieval/grounding can still be important.

---

# 20. Fine-tuning kab karni chahiye?

Fine-tuning useful ho sakti hai jab:

```text
Repeated task
Specific output format
Specific writing style
Domain-specific language
Special classification task
Consistent behavior
Instruction following
```

Example:

```text
Input:
Customer complaint

Output:
Category + priority + sentiment
```

Agar tumhare paas thousands of examples hain, fine-tuning useful approach ho sakti hai.

---

# 21. Fine-tuning kab nahi karni chahiye?

Suppose tum bas chahte ho:

> "Model ko meri latest company documents pata hon."

Automatically fine-tuning best answer nahi hai.

Because documents:

```text
change frequently
```

Instead:

```text
RAG
```

may be more appropriate.

Also, fine-tuning generally shouldn't be the first solution if:

```text
Prompt engineering solves the problem
```

A common progression is:

```text
Prompting
   ↓
RAG
   ↓
Fine-tuning
```

Not always, but this is a useful engineering thought process.

---

# 22. Prompting vs RAG vs Fine-tuning

This is a very useful interview comparison.

| Technique   | Main purpose                             |
| ----------- | ---------------------------------------- |
| Prompting   | Tell model how to respond                |
| RAG         | Give model relevant external knowledge   |
| Fine-tuning | Teach specialized behavior/task patterns |

Easy example:

### Prompting

```text
"Answer in 3 bullet points."
```

### RAG

```text
Here are the company's latest policy documents.
```

### Fine-tuning

```text
Thousands of examples showing exactly how
company-specific support responses should be generated.
```

---

# 23. Does fine-tuning create a completely new model?

Not usually in the way beginners imagine.

Usually:

```text
Pretrained Base Model
        ↓
Further Training
        ↓
Specialized Model / Adapter
```

The base capabilities remain, but the model's parameters or attached adapters are adjusted toward the target behavior.

---

# 24. One important concept: Catastrophic forgetting

Suppose model already knows:

```text
English
Programming
General reasoning
```

You fine-tune it too aggressively on a narrow dataset:

```text
Only legal documents
```

The model can lose some performance on capabilities that were previously strong.

This is called **catastrophic forgetting**.

That's one reason careful data selection, training setup, and evaluation matter.

---

# 25. Fine-tuning ≠ training from scratch

Very important:

```text
Training from scratch
```

means:

```text
Random/initial parameters
       ↓
Huge dataset
       ↓
Train entire model
```

Very expensive.

Fine-tuning:

```text
Already pretrained model
       ↓
Smaller task-specific dataset
       ↓
Further training
```

Much more practical.

---

# 26. A complete mental picture

Remember this:

```text
                  AI MODEL JOURNEY

Huge general dataset
        ↓
   PRETRAINING
        ↓
   Base LLM
        │
        ├──────── Prompting
        │             ↓
        │      Better instructions
        │
        ├──────── RAG
        │             ↓
        │      External knowledge
        │
        └──────── Fine-tuning
                      ↓
              Specialized behavior
                      ↓
             Task-specific model
```

And these can be combined:

```text
Fine-tuned LLM
      +
     RAG
      +
Good prompting
      ↓
Production GenAI System
```

---

# 27. Interview mein "Fine-tuning" kaise explain karna hai?

### 30-second answer

Fine-tuning is the process of taking a pretrained language model and training it further on a smaller, task-specific dataset so that it performs better for a particular task, domain, style, or output format.

For example, a general LLM can be fine-tuned on thousands of customer-support conversations so that it learns the company's preferred response style and behavior.

During fine-tuning, the model learns from input-output examples and updates its parameters to reduce the training loss. In modern LLMs, parameter-efficient techniques such as LoRA and QLoRA are often used to reduce GPU memory, training cost, and computation.

Unlike RAG, which provides external information to the model at inference time, fine-tuning changes the model's learned behavior.

### Interviewer agar pooche: "When would you use fine-tuning?"

I would consider fine-tuning when the model needs consistent domain-specific behavior, a specific response format, a particular communication style, or improved performance on a repeated task.

For frequently changing or private factual information, I would generally consider RAG because the knowledge can be retrieved and updated without retraining the model.

### One-line answer to remember

> **RAG teaches the model what information to look at; fine-tuning teaches the model how to behave on a specific task.**

Bas is line ko properly samajh liya, to **fine-tuning ka core concept clear hai**.

Next level par tumhe **SFT → LoRA → QLoRA → dataset preparation → training → evaluation** samajhna chahiye, kyunki GenAI Engineer interviews mein ye fine-tuning ka practical side hota hai.

Absolutely. Ab hum **Fine-tuning ka practical side** step-by-step samajhte hain:

**SFT → LoRA → QLoRA → Dataset → Training → Evaluation**

Main pehle simple example, phir technical explanation dunga.

---

# 1. SFT — Supervised Fine-Tuning

SFT ka full form hai:

> **Supervised Fine-Tuning**

Ye fine-tuning ka sabse common form hai.

### Simple example

Maan lo ek teacher student ko 1,000 questions deta hai:

```text
Question → Correct Answer
```

Example:

```text
Q: What is Python?
A: Python is a high-level programming language.
```

Student in examples se seekhta hai.

LLM mein bhi:

```text
Input → Expected Output
```

examples diye jaate hain.

### Example

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is RAG?"
    },
    {
      "role": "assistant",
      "content": "RAG stands for Retrieval-Augmented Generation..."
    }
  ]
}
```

Bahut saare examples:

```text
Example 1
Example 2
Example 3
...
Example 10,000
```

Model in examples par train hota hai.

### Main idea

> **SFT teaches the model the desired input-output behavior using labeled examples.**

---

# 2. LoRA

Ab problem aati hai.

Suppose model bahut bada hai:

```text
70B parameters
```

Agar hum saare parameters train karein, to:

```text
GPU memory ↑
Training cost ↑
Training time ↑
```

Isliye LoRA use kar sakte hain.

**LoRA = Low-Rank Adaptation**

### Simple life example

Maan lo tum ek already-trained employee ko new company mein laaye.

Us employee ko completely retrain nahi karna.

Bas:

```text
Company-specific training
+
Small additional knowledge/behavior
```

de rahe ho.

LLM mein bhi base model ko mostly freeze kar dete hain.

```text
Base Model
   ↓
Frozen ❄️
   
Small LoRA parameters
   ↓
Trainable 🔥
```

Matlab:

> Original model ko mostly unchanged rakhkar small additional parameters train karte hain.

---

# 3. LoRA technically kya karta hai?

Suppose original weight matrix hai:

```text
W
```

Normal fine-tuning mein:

```text
W → W'
```

Matlab original weights directly update.

LoRA mein conceptually:

```text
W' = W + ΔW
```

Aur `ΔW` ko directly huge matrix ki tarah learn karne ke instead, low-rank matrices se represent karte hain:

```text
ΔW = A × B
```

Jahan:

```text
A = small matrix
B = small matrix
```

Isliye trainable parameters dramatically reduce ho sakte hain.

Tumhe interview mein mathematics explain karne ki zarurat usually nahi hoti unless interviewer deep dive kare.

---

# 4. QLoRA

Ab LoRA ke upar next idea:

**QLoRA = Quantized LoRA**

Do techniques combine hoti hain:

```text
Quantization
      +
LoRA
```

### Quantization kya hai?

Model ke numbers ko lower precision mein represent karna.

Conceptually:

```text
FP32
 ↓
FP16 / BF16
 ↓
INT8 / 4-bit
```

Lower precision se memory usage reduce ho sakta hai.

So:

```text
Large model
   ↓
Quantize it
   ↓
Attach LoRA adapters
   ↓
Train adapters
```

Isse large model fine-tuning comparatively resource-efficient ho sakti hai.

### Easy memory trick

```text
LoRA
= Train fewer parameters

QLoRA
= Quantize model + train LoRA adapters
```

---

# 5. SFT vs LoRA vs QLoRA

Important:

Ye same category ke competing words nahi hain.

Think like this:

```text
SFT
= WHAT training objective/process are we doing?

LoRA
= HOW are we efficiently updating the model?

QLoRA
= HOW are we doing LoRA with a quantized model?
```

For example:

```text
SFT + LoRA
```

bilkul valid combination hai.

Aur:

```text
SFT + QLoRA
```

bhi valid hai.

---

# 6. Dataset preparation

Fine-tuning ka actual success heavily dataset quality par depend karta hai.

Suppose tum customer-support model bana rahe ho.

Bad dataset:

```text
User: refund?
Assistant: okay
```

Bahut vague.

Better:

```text
User:
I want a refund for my order because the product was damaged.

Assistant:
I'm sorry the product arrived damaged. To request a refund, please open
Orders → Select the product → Request Refund.
```

Model ko clear pattern dikhta hai.

---

# 7. Dataset mein kya hona chahiye?

Good fine-tuning dataset generally:

```text
Relevant
Correct
Consistent
Diverse
High-quality
```

Example:

```text
Input                         Output
--------------------------------------------------
Refund request                Proper refund response
Late delivery                 Delivery response
Wrong product                 Replacement response
Damaged product               Damage/refund response
...
```

Important point:

> **More data is not automatically better. High-quality and representative data matters.**

---

# 8. Train / Validation split

Suppose tumhare paas:

```text
10,000 examples
```

Sabse important mistake hogi:

> Saare 10,000 examples training mein daal dena aur phir unhi par test karna.

Instead:

```text
Training data
Validation data
Test data
```

Example:

```text
80% → Training
10% → Validation
10% → Test
```

Exact ratio project ke according change ho sakta hai.

### Training set

Model actually learns from this.

### Validation set

Training ke during model ko monitor/tune karne ke liye.

### Test set

Final unseen performance check karne ke liye.

---

# 9. Training process

High-level flow:

```text
Dataset
   ↓
Tokenizer
   ↓
Base Model
   ↓
Forward Pass
   ↓
Prediction
   ↓
Calculate Loss
   ↓
Backpropagation
   ↓
Update trainable parameters
   ↓
Repeat
```

Ye process multiple batches aur epochs tak hota hai.

---

# 10. Tokenization

LLM directly sentences ko understand nahi karta.

Text ko tokens mein convert kiya jata hai.

Example:

```text
"I love Python"
```

roughly:

```text
["I", "love", "Python"]
```

Actual tokenizer words ko subwords/tokens mein split kar sakta hai.

Then:

```text
tokens
 ↓
token IDs
 ↓
model
```

Example conceptually:

```text
"I love Python"
      ↓
[40, 821, 19342]
```

Exact IDs tokenizer par depend karte hain.

---

# 11. Loss kya hota hai?

Suppose expected answer:

```text
Python is a programming language.
```

Model generated:

```text
Python is a programming tool.
```

Model ka prediction aur target compare hota hai.

Difference ko training loss se represent kiya jata hai.

```text
Prediction
   +
Target
   ↓
Loss
   ↓
Backpropagation
   ↓
Parameter update
```

Goal:

> **Training ke through loss ko reduce karna.**

---

# 12. Epoch

Suppose:

```text
10,000 training examples
```

Model ne poora dataset ek baar dekha:

```text
1 epoch
```

Twice:

```text
2 epochs
```

Three times:

```text
3 epochs
```

But:

```text
More epochs ≠ always better
```

Too much training can lead to overfitting.

---

# 13. Fine-tuning ke important hyperparameters

Interview ke liye inko know karo:

### Learning rate

Weight updates kitne bade honge.

```text
Too high → unstable
Too low  → slow learning
```

### Batch size

Ek step mein kitne training examples process hote hain.

### Epochs

Dataset ko kitni baar process karna hai.

### Weight decay

Regularization mein help karta hai aur overfitting control karne mein useful ho sakta hai.

### Gradient accumulation

Limited GPU memory hone par effectively larger batch simulate karne mein help karta hai.

---

# 14. Evaluation

Training complete ho gayi.

Ab question:

> "Model actually better hua ya nahi?"

Sirf training loss dekhna enough nahi hai.

Tum compare kar sakte ho:

```text
Base model
     vs
Fine-tuned model
```

on unseen test examples.

---

# 15. Evaluation mein kya check karna hai?

Task ke according metrics change hoti hain.

For classification:

```text
Accuracy
Precision
Recall
F1-score
```

For generation:

```text
Loss / perplexity
Task-specific evaluation
Human evaluation
LLM-as-judge (with care)
Exact-match / structured-output checks
```

RAG systems ke liye alag retrieval and answer-quality metrics bhi use hote hain.

---

# 16. Real practical example

Suppose tum **customer-support AI** bana rahe ho.

Tumhare paas:

```text
20,000 support conversations
```

Workflow:

```text
20,000 conversations
        ↓
Clean data
        ↓
Convert to chat format
        ↓
Train/validation/test split
        ↓
Choose base LLM
        ↓
SFT
        ↓
Use LoRA/QLoRA
        ↓
Train
        ↓
Evaluate
        ↓
Deploy
```

Let's say before fine-tuning:

```text
Base model:
Generic responses
```

After fine-tuning:

```text
Better adherence to company's
response format and support style
```

That's the goal.

---

# 17. Very important: Fine-tuning doesn't necessarily "teach facts"

Suppose you have:

```text
Company policy:
Employees get 24 leaves.
```

And next month it changes:

```text
24 → 30
```

A fine-tuned model isn't automatically updated just because your company document changed.

That's why for dynamic knowledge:

```text
RAG
```

is often useful.

Think:

```text
Fine-tuning → behavior
RAG         → external/current knowledge
Prompting   → instructions
```

---

# 18. One analogy that makes everything easy

Imagine **you hire a chef**.

### Pretraining

Chef already knows:

```text
Indian food
Chinese food
Italian food
Baking
Knife skills
```

### SFT

You give chef:

```text
1,000 examples of your restaurant's dishes
```

and say:

> "Prepare dishes this way."

### LoRA

Instead of completely retraining chef, give them a **small specialized training module**.

### QLoRA

Same idea, but you also make the underlying setup more memory/resource efficient.

### RAG

During cooking you give chef today's menu and ingredient availability.

### Prompting

You tell chef:

> "Make the dish less spicy and serve it in 10 minutes."

This analogy is very close to how these concepts fit together.

---

# 19. Interview question: "Explain SFT, LoRA and QLoRA"

Perfect answer:

SFT stands for Supervised Fine-Tuning. It trains a pretrained model on labeled input-output examples so the model learns a specific task, behavior, style, or output format.

LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning technique. Instead of updating most of the original model parameters, it keeps the base model largely frozen and trains a small set of adapter parameters, which reduces memory and computational requirements.

QLoRA extends this idea by quantizing the base model, typically to a lower-bit representation, and then training LoRA adapters on top of it. This further reduces memory requirements and makes fine-tuning large language models more resource-efficient.

So, SFT describes the supervised training approach, while LoRA and QLoRA describe more efficient ways of adapting the model.

---

# 20. Most important interview question: "Fine-tuning vs RAG?"

Fine-tuning and RAG solve different problems.

Fine-tuning updates the model's learned parameters or adapters so that it performs better on a specific task, behavior, style, or output format.

RAG does not retrain the model. Instead, it retrieves relevant external information at inference time and provides that information as context to the model.

For example, if I want a model to consistently classify customer complaints in a specific format, fine-tuning can be useful. If I want it to answer questions using frequently changing company documents, RAG is usually more appropriate.

They can also be combined in a production system.

### Ek line mein sab yaad rakho:

```text
Prompting  → Tell the model what to do
RAG        → Give the model relevant information
SFT        → Train the model on desired examples
LoRA       → Fine-tune efficiently
QLoRA      → Quantized model + LoRA
```

**Next logical step:** actual hands-on mein ek small **LoRA/QLoRA fine-tuning project** banana, jisme dataset se lekar training aur evaluation tak poora flow dekhenge.
