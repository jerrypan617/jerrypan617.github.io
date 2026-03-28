// Tailwind CSS Configuration — OpenAI / Grok–style dark product UI
tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                'accent-color': '#10a37f',
                surface: {
                    DEFAULT: '#0a0a0a',
                    raised: '#141414',
                    overlay: '#1a1a1a',
                },
            },
            fontFamily: {
                sans: ['"Inter"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
        }
    }
}
