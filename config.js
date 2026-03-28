const siteConfig = {
    personal: {
        name: "Xu_Pan",
        title: "PhD_STUDENT [Computer Science]",
        email: "pan2004xu@gmail.com",
        phone: "(+86) 13732230785",
        bootMessage: "Computer science · Xi'an Jiaotong University",
        social: {
            x: "https://x.com/Jerry1865942",
            huggingface: "https://huggingface.co/JERRYPAN617",
            reddit: "https://www.reddit.com/user/Background-Pilot-288/"
        }
    },

    profile: {
        education: [
            {
                degree: "PhD",
                major: "CS",
                university: "Xi'an JiaoTong University",
                period: "2026.09-Future"
            },
            {
                degree: "Bachelor",
                major: "CS",
                university: "Hefei University of Technology",
                period: "2022.09-2026.07"
            }
        ],
        focus: "LLM Memory Augmentation & Vision-Text Alignment"
    },

    skills: {
        programming: [
            "Python",
            "C++",
            "C",
            "x86 Assembly"
        ],
        frameworks: [
            "PyTorch",
            "OpenCV",
            "Numpy",
            "Transformers",
            "TRL"
        ],
        domains: [
            "LLM Memory Augmentation",
            "Vision-Text Alignment",
            "Reinforcement Learning from Human Feedback"
        ]
    },

    projects: [
        {
            type: "RESEARCH",
            name: "AFDNet",
            subtitle: "Visual Algorithms for Single Image Rain Removal",
            description: "Developed deep learning algorithms to effectively remove visual degradation features caused by rain, enhancing vision systems.",
            technologies: ["Python", "MindSpore", "OpenCV"],
            github: "https://github.com/jerrypan617/DerainUNet-MindSpore",
            paper: ""
        },
        {
            type: "RESEARCH",
            name: "CF-CAM",
            subtitle: "Gradient Perturbation Mitigation",
            description: "Improved gradient-based CAM algorithms by mitigating noise through clustering and filtering, significantly enhancing deep learning model interpretability.",
            technologies: ["Python", "PyTorch", "OpenCV"],
            github: "https://github.com/jerrypan617/CF-CAM",
            paper: "https://arxiv.org/abs/2504.00060"
        },
        {
            type: "PROJECT",
            name: "PXOS",
            subtitle: "A Mini 32-bit x86 kernel",
            description: "A 32-bit x86 operating system kernel written in C and x86 asm, featuring an architecture with process & memory management, file system, and an interactive shell.",
            technologies: ["C", "Assembly", "Makefile"],
            github: "https://github.com/jerrypan617/PXOS",
        },
        {
            type: "PROJECT",
            name: "Qwen2.5 Fine-Tuning",
            subtitle: "Instruct Model with GRPO",
            description: "Fine-tuned LLM using GRPO to implement Chain-of-Thought reasoning, achieving a ~10% increase in mathematical reasoning accuracy.",
            technologies: ["Python", "Transformers", "TRL", "PyTorch"],
            github: "",
            paper: ""
        },
        {
            type: "PROJECT",
            name: "Med-VQA-BLIP",
            subtitle: "BLIP Fine-Tuning",
            description: "A fine-tuned BLIP-VQA model for medical pathology image question answering with open-ended text generation.",
            technologies: ["Python", "Transformers","Peft", "PyTorch"],
            github: "https://github.com/jerrypan617/Medical-VQA-BLIP",
        },
        {
            type: "PROJECT",
            name: "Simple RAG",
            subtitle: "Two-phase RAG system for LLM reasoning.",
            description: "A 2-phase (Retrieve + Rerank) RAG system implementation.",
            technologies: ["Python", "Sentence_Transformers", "PyTorch", "faiss-cpu"],
            github: "https://github.com/jerrypan617/Simple-RAG",
        }
    ],

    achievements: [
        {
            title: "China National Scholarship",
            subtitle: "",
            date: "2024/01",
            color: "neon-pink"
        },
        {
            title: "Contemporary Undergraduate Mathematical Contest in Modeling",
            subtitle: "National 2nd prize",
            date: "2024/11",
            color: "neon-lime"
        },
        {
            title: "COMAP Mathematical Contest in Modeling",
            subtitle: "Meritorious Winner",
            date: "2024/05",
            color: "neon-cyan"
        }
    ],

    footer: {
        systemId: "© Xu Pan",
        designPattern: ""
    },
    
    // Blog posts
    blogs: [
        {
            id: "1",
            title: "LLM Memory Augmentation Techniques",
            subtitle: "Enhancing large language models with external memory systems",
            date: "2024-01-15",
            content: `# LLM Memory Augmentation

Large Language Models (LLMs) have revolutionized natural language processing, but they still face limitations in long-term memory and knowledge retention. In this post, I'll explore various techniques for augmenting LLM memory.

## 1. Retrieval-Augmented Generation (RAG)

RAG combines retrieval of external documents with generation:

\`\`\`python
def rag_pipeline(query):
    # Retrieve relevant documents
    documents = retriever.retrieve(query)
    # Generate response using retrieved documents
    response = llm.generate(query, documents)
    return response
\`\`\`

## 2. Attention Mechanisms

The transformer architecture uses self-attention to process input sequences:

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

This allows the model to focus on relevant parts of the input when generating output.

## 3. Memory Networks

Memory networks introduce explicit memory components that can be read from and written to:

- External memory banks
- Differentiable access mechanisms
- Long-term knowledge storage

## 4. Recent Advances

Recent work has focused on:

- Efficient retrieval algorithms
- Dynamic memory management
- Multi-modal memory integration

## Conclusion

Memory augmentation is a promising direction for improving LLMs, enabling them to handle longer contexts and retain knowledge more effectively.

---

*References:*
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Memory-Augmented Neural Networks
- The Transformer Architecture`,
            tags: ["LLM", "Memory", "AI", "Research"]
        },
        {
            id: "2",
            title: "Vision-Text Alignment in Multi-modal Models",
            subtitle: "Bridging the gap between visual and textual representations",
            date: "2024-02-01",
            content: `# Vision-Text Alignment

Multi-modal models that can process both images and text have become increasingly important. A key challenge is aligning visual and textual representations.

## Contrastive Learning

Contrastive learning techniques like CLIP have shown success in aligning vision and text:

$$
L = -\log\frac{e^{sim(I, T)/\tau}}{\sum_{t=1}^N e^{sim(I, T_t)/\tau}} - \log\frac{e^{sim(I, T)/\tau}}{\sum_{i=1}^N e^{sim(I_i, T)/\tau}}
$$

## Cross-Modal Attention

Cross-modal attention mechanisms allow the model to attend to relevant parts of one modality when processing another:

- Image attention over text
- Text attention over image regions
- Multi-head cross-modal attention

## Evaluation Metrics

Common metrics for vision-text alignment include:

- Zero-shot classification accuracy
- Text-to-image retrieval
- Image-to-text retrieval

## Future Directions

- Fine-grained alignment at the region level
- Dynamic attention mechanisms
- Incorporating structured knowledge`,
            tags: ["Multi-modal", "Vision", "NLP", "AI"]
        }
    ]
};

