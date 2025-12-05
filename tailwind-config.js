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
                mono: ['"Courier Prime"', '"Courier New"', 'Courier', 'monospace'],
            },
            boxShadow: {
                'neon-cyan': '0 0 5px #00f3ff, 0 0 10px #00f3ff',
                'neon-pink': '0 0 5px #ff00ff, 0 0 10px #ff00ff',
                'neon-lime': '0 0 5px #ccff00, 0 0 10px #ccff00',
            },
            animation: {
                'scanline': 'scanline 8s linear infinite',
                'flicker': 'flicker 0.15s infinite',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'blink': 'blink 1s step-end infinite',
            },
            keyframes: {
                scanline: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(100%)' }
                },
                flicker: {
                    '0%': { opacity: '0.97' },
                    '5%': { opacity: '0.9' },
                    '10%': { opacity: '0.97' },
                    '15%': { opacity: '1' },
                    '100%': { opacity: '0.98' }
                },
                blink: {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0' }
                }
            }
        }
    }
}

