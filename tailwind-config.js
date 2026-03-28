// Tailwind CSS Configuration
tailwind.config = {
    theme: {
        extend: {
            colors: {
                'neon-cyan': '#00f3ff',
                'neon-pink': '#ff00ff',
                'neon-lime': '#ccff00',
                'retro-bg': '#050505',
                'retro-dark': '#0a0a0a',
            },
            fontFamily: {
                sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
        }
    }
}
