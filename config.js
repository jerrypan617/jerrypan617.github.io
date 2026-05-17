const siteConfig = {
    /**
     * 站点根 URL（无末尾斜杠），与各页 canonical/OG 保持一致。
     * 若使用自定义域名或 Project Pages，请同时改 HTML/feed 中的绝对链接。
     */
    site: {
        url: 'https://jerrypan617.github.io',
        /** Open Graph / Twitter 默认图，须为 https 绝对地址 */
        ogImage: 'https://jerrypan617.github.io/resources/my_face.jpg'
    },

    /** 填入 Google Analytics 4 衡量 ID（G-xxxxxxxxxx）后启用；留空则不加载 gtag */
    analytics: {
        ga4MeasurementId: 'G-5C9YNPZ5SD'
    },

    personal: {
        name: "Xu_Pan",
        email: "pan2004xu@gmail.com",
        phone: "(+86) 13732230785",
        photo: "resources/my_face.jpg",
        social: {
            github: "https://github.com/jerrypan617",
            googleScholar: "https://scholar.google.com/citations?user=lSA9xFoAAAAJ&hl=zh-CN",
            huggingface: "https://huggingface.co/JERRYPAN617",
            reddit: "https://www.reddit.com/user/Background-Pilot-288/"
        }
    },

    profile: {
        education: [
            {
                logo: "resources/xian-jiaotong-university-logo-1024px.png",
                title: "Ph.D. in Computer Science and Technology",
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

    /** 研究兴趣标签（页面展示为 Research interests） */
    skills: {
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
            paperTitle:
                "Adaptive Feature extraction and Deformable Sampling Network for Rain Removal",
            authors: [
                { name: "X. Pan" },
                { name: "Y. Yu", corresponding: true },
                { name: "Y. Liang" },
                { name: "J. Yi" },
                { name: "Q. Lu" },
            ],
            github: "https://github.com/jerrypan617/DerainUNet-MindSpore",
            paper: "",
        },
        {
            type: "RESEARCH",
            paperTitle:
                "CF-CAM: Cluster Filter Class Activation Mapping for Reliable Gradient-Based Interpretability",
            authors: [
                { name: "X. Pan", coFirst: true },
                { name: "H. He", coFirst: true },
                { name: "Y. Yao", corresponding: true },
            ],
            github: "https://github.com/jerrypan617/CF-CAM",
            paper: "https://arxiv.org/abs/2504.00060",
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

};

