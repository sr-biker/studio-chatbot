"""Hand-labeled eval questions grounded in data/faq.md, used by both the RAGAS suite
(faithfulness/answer-relevancy/context-precision) and the openai-evals-style exact-match
suite. Keep `ground_truth` short and unambiguous -- these are graded by an LLM judge
(RAGAS) or containment (openai-evals-style), not free-form comparison."""

FAQ_EVAL_CASES = [
    {
        "question": "Do I need a membership to attend classes?",
        "ground_truth": "Most classes require membership, but non-members may buy a day pass or register for eligible drop-in classes.",
    },
    {
        "question": "Can I cancel my class registration and get a refund?",
        "ground_truth": "Yes, cancellations made at least 24 hours before the class or event receive a full credit or refund, depending on the event policy.",
    },
    {
        "question": "How early should I arrive for a class?",
        "ground_truth": "Arrive 10 to 15 minutes before your scheduled class or event to check in and get settled.",
    },
    {
        "question": "Do I need to bring my own yoga mat?",
        "ground_truth": "You're welcome to bring your own mat, but mats are also available to borrow or rent.",
    },
    {
        "question": "What's the difference between Pilates and yoga?",
        "ground_truth": "Pilates focuses on core strength, posture, and controlled movement, while yoga emphasizes flexibility, balance, and mindfulness.",
    },
    {
        "question": "What happens if a class I want is full?",
        "ground_truth": "There is a waitlist; if a spot opens up you're automatically notified and enrolled if you confirm within the required timeframe.",
    },
    {
        "question": "How much does a membership cost per month with a 1-year commitment?",
        "ground_truth": "$69 per month with a 1-year commitment ($79/month month-to-month, $59/month with a 2-year commitment).",
    },
]
