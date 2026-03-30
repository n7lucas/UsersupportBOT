## 🧠 Sentiment Classifier: Performance & Architectural Justification
While the Intent Classifier achieves near-perfect accuracy (99.5%), the Sentiment Classifier operates at a ~66% overall accuracy. In the context of our production architecture, this is considered a highly successful baseline. Here is the data science and business rationale behind this metric.

1. The Subjectivity of Human Emotion
Unlike intent prediction (which is highly objective and keyword-driven, e.g., "cancel" = cancel_order), sentiment analysis is inherently subjective.

The target labels (Frustrated, Neutral, Polite, Urgent) were distilled using a Zero-Shot LLM (gpt-4o-mini).

In a 4-class subjective problem, a random guess yields 25% accuracy. Achieving ~66% using a blazing-fast, lightweight Logistic Regression model running on all-MiniLM-L6-v2 embeddings proves the model is capturing deep emotional semantics without the latency of a heavy neural network.

2. Overcoming Target Leakage and LLM Bias
During the data labeling phase, the LLM exhibited a known bias: it aggressively categorized customer support tickets as Urgent (accounting for nearly 50% of the dataset).
Initially, the Logistic Regression model exploited this imbalance, resulting in a slightly higher overall accuracy, but a poor Recall for critical minority classes:

Initial 'Frustrated' Recall: 0.51 (The system was missing half of the angry customers).

3. Strategic Tradeoff: Recall > Overall Accuracy
In a customer support pipeline, missing a frustrated customer is a critical business failure that leads to churn. To fix this, we applied class_weight='balanced' to the Logistic Regression model.

The Final Results:

Overall Accuracy stabilized at 66.35%.

'Frustrated' Recall skyrocketed to 0.78. * 'Polite' Recall improved to 0.61.

The model stopped lazily predicting 'Urgent' as a default fallback.

By deliberately sacrificing a marginal amount of overall precision, we built a much fairer, highly sensitive "EQ Brain" for the system. When the Planner node receives a Frustrated flag, it can now confidently instruct the downstream LLM Agent to prioritize empathy and de-escalation.