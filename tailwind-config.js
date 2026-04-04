// Tailwind CSS Configuration — light UI
tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                'accent-color': '#0d9f7a',
                surface: {
                    DEFAULT: '#ffffff',
                    raised: '#fafafa',
                    overlay: '#f4f4f5',
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
