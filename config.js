const siteConfig = {
    personal: {
        name: "Xu_Pan",
        title: "PhD_STUDENT [Computer Science]",
        email: "pan2004xu@gmail.com",
        phone: "(+86) 13732230785",
        bootMessage: "Computer science · Xi'an Jiaotong University",
        photo: "resources/my_face.jpg",
        social: {
            googleScholar: "https://scholar.google.com/citations?user=lSA9xFoAAAAJ&hl=zh-CN",
            huggingface: "https://huggingface.co/JERRYPAN617",
            reddit: "https://www.reddit.com/user/Background-Pilot-288/"
        }
    },

    profile: {
        education: [
            {
                logo: "resources/xian-jiaotong-university-logo-1024px.png",
                title: "Ph.D. in Computer Science",
                institution: "Xi'an Jiaotong University",
                unit: "School of Computer Science and Technology",
                location: "Xi'an, Shaanxi, China",
                period: "Sep 2026 — present"
            },
            {
                logo: "resources/hefei-university-of-technology-logo-1024px.png",
                title: "B.Eng. in Computer Science and Technology",
                institution: "Hefei University of Technology",
                unit: "School of Computer Science and Information Engineering",
                location: "Hefei, Anhui, China",
                period: "Sep 2022 — Jul 2026"
            }
        ]
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
            "Latent Reasoning",
            "LLM Memory",
            "RLHF",
            "XAI",
            "Image Reconstruction"
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

    /** Markdown 文件名列表（相对 /blogs/）。GitHub Pages 无目录索引时以此为准；本地也可继续用目录扫描作补充。 */
    blogFiles: [
        "attention-sink-analysis.md"
    ]
};

