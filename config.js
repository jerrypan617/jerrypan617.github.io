const siteConfig = {
    personal: {
        name: "Xu_Pan",
        title: "PhD_STUDENT [Computer Science]",
        email: "pan2004xu@gmail.com",
        phone: "(+86) 13732230785",
        bootMessage: "> SYSTEM_BOOT_SEQUENCE_INITIATED...",
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
        systemId: "XP-2025",
        designPattern: "CASSETTE_FUTURISM"
    }
};

