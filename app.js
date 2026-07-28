function getPreferredTheme() {
    let storedTheme = null;
    try {
        storedTheme = localStorage.getItem('theme');
    } catch (error) {}
    if (storedTheme === 'light' || storedTheme === 'dark') return storedTheme;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
    const resolvedTheme = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.style.colorScheme = resolvedTheme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolvedTheme === 'dark' ? '#141513' : '#f7f6f1');
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
        button.setAttribute('aria-label', `Switch to ${resolvedTheme === 'dark' ? 'light' : 'dark'} mode`);
        button.setAttribute('aria-pressed', String(resolvedTheme === 'dark'));
    });
}

function toggleTheme() {
    const nextTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    try {
        localStorage.setItem('theme', nextTheme);
    } catch (error) {}
    applyTheme(nextTheme);
}

applyTheme(getPreferredTheme());

function themeToggleHtml() {
    return `
        <button type="button" data-theme-toggle onclick="toggleTheme()" class="theme-toggle pressable" aria-label="Switch color mode" aria-pressed="false">
            <svg class="theme-icon theme-icon-moon" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12.8A8.5 8.5 0 1111.2 3a6.6 6.6 0 009.8 9.8z"/>
            </svg>
            <svg class="theme-icon theme-icon-sun" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4V2m0 20v-2m8-8h2M2 12h2m14.95-6.95l1.42-1.42M3.63 20.37l1.42-1.42m0-13.9L3.63 3.63m16.74 16.74l-1.42-1.42"/>
                <circle cx="12" cy="12" r="4" stroke-width="2"/>
            </svg>
            <span class="theme-label"></span>
        </button>
    `;
}

function renderThemeDock() {
    let dock = document.getElementById('theme-dock');
    if (!dock) {
        dock = document.createElement('div');
        dock.id = 'theme-dock';
        dock.className = 'theme-dock';
        document.body.appendChild(dock);
    }
    dock.innerHTML = themeToggleHtml();
}

// 渲染左侧边栏：头像、基本信息、外链与快速导航（首页 / 博客共用布局）
function renderHeader() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    const { personal } = siteConfig;
    const displayName = personal.name.replace(/_/g, ' ');

    const avatarCompact =
        'sidebar-avatar sidebar-avatar-compact relative block shrink-0 overflow-hidden rounded-sm border border-zinc-200/70 bg-zinc-100 w-8 h-8';
    const avatarFull =
        'sidebar-avatar sidebar-avatar-full relative block shrink-0 overflow-hidden rounded-sm border border-zinc-200/70 bg-zinc-100 transition-opacity duration-300';

    const avatarImg = personal.photo
        ? `<img src="${personal.photo}" alt="${displayName}" width="295" height="413" decoding="async" class="absolute inset-0 h-full w-full object-cover object-top" onerror="this.classList.add('hidden'); this.nextElementSibling.classList.remove('hidden')" />
           <span class="hidden absolute inset-0 bg-zinc-200/80 flex items-center justify-center text-zinc-600 font-semibold text-xs" aria-hidden="true">XP</span>`
        : `<span class="absolute inset-0 flex items-center justify-center text-zinc-600 font-semibold text-xs">XP</span>`;

    const socialIcons = `
        ${personal.social.github ? `
            <a href="${personal.social.github}" target="_blank" rel="noopener noreferrer" class="pressable sidebar-social-link w-7 h-7 flex items-center justify-center rounded-sm text-zinc-500 hover:text-emerald-700 transition-all" title="GitHub" aria-label="GitHub profile">
                <svg class="w-[17px] h-[17px]" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            </a>
        ` : ''}
        ${personal.social.googleScholar ? `
            <a href="${personal.social.googleScholar}" target="_blank" rel="noopener noreferrer" class="pressable sidebar-social-link w-7 h-7 flex items-center justify-center rounded-sm text-zinc-500 hover:text-emerald-700 transition-all" title="Google Scholar" aria-label="Google Scholar profile">
                <svg class="w-[17px] h-[17px]" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l9-5-9-5-9 5 9 5z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/></svg>
            </a>
        ` : ''}
        ${personal.social.huggingface ? `
            <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" class="pressable sidebar-social-link w-7 h-7 flex items-center justify-center rounded-sm text-zinc-500 hover:text-emerald-700 transition-all" title="HuggingFace" aria-label="HuggingFace profile">
                <span style="font-size:17px;line-height:1">🤗</span>
            </a>
        ` : ''}
    `;

    const socialRows = `
        ${personal.social.github ? `
            <a href="${personal.social.github}" target="_blank" rel="noopener noreferrer" class="link-minimal group flex items-center gap-1.5 transition-colors hover:text-emerald-600">
                <div class="w-5 h-5 flex items-center justify-center shrink-0 transition-colors">
                    <svg class="w-3.5 h-3.5 text-zinc-500 group-hover:text-emerald-600 transition-colors" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                </div>
                <span class="pb-0.5">GitHub</span>
            </a>
        ` : ''}
        ${personal.social.googleScholar ? `
            <a href="${personal.social.googleScholar}" target="_blank" rel="noopener noreferrer" class="link-minimal group flex items-center gap-1.5 transition-colors hover:text-emerald-600">
                <div class="w-5 h-5 flex items-center justify-center shrink-0 transition-colors">
                    <svg class="w-3.5 h-3.5 text-zinc-500 group-hover:text-emerald-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l9-5-9-5-9 5 9 5z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/></svg>
                </div>
                <span class="pb-0.5">Google Scholar</span>
            </a>
        ` : ''}
        ${personal.social.huggingface ? `
            <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" class="link-minimal group flex items-center gap-1.5 transition-colors hover:text-emerald-600">
                <div class="w-5 h-5 flex items-center justify-center shrink-0 transition-colors">
                    <span class="text-[13px] leading-none" aria-hidden="true">🤗</span>
                </div>
                <span class="pb-0.5">HuggingFace</span>
            </a>
        ` : ''}
    `;

    const educationHtml = (siteConfig.profile?.education || [])
        .map((edu) => {
            const institution = edu.institution || edu.university || '';
            const title = edu.title || `${edu.degree || ''}${edu.major ? ` in ${edu.major}` : ''}`.trim();
            const altLogo = institution || title;
            return `
                <div class="sidebar-education-item">
                    ${edu.logo ? `
                        <div class="sidebar-education-logo" aria-hidden="true">
                            <img src="${edu.logo}" alt="${altLogo}" width="48" height="48" loading="lazy" decoding="async" />
                        </div>
                    ` : ''}
                    <div class="sidebar-education-copy">
                        <p class="sidebar-education-title">${title}</p>
                        <p class="sidebar-education-meta">${institution}</p>
                        <p class="sidebar-education-period">${edu.period}</p>
                    </div>
                </div>
            `;
        })
        .join('');

    sidebar.innerHTML = `
        <!-- ===== Mobile layout (< lg) ===== -->
        <div class="flex flex-col lg:hidden">
            <div class="flex items-center justify-between gap-2 pr-14">
                <div class="flex items-center gap-2 min-w-0">
                    <span class="${avatarCompact}">${avatarImg}</span>
                    <div class="min-w-0">
                        <a href="index.html" class="text-sm font-bold text-zinc-950 tracking-tight hover:text-zinc-600 transition-colors">${displayName}</a>
                        <p class="text-[9px] font-semibold tracking-[0.12em] uppercase text-zinc-500 mt-0.5">Research profile</p>
                    </div>
                </div>
                <div class="flex items-center gap-0.5 shrink-0 -mr-1.5">
                    ${socialIcons}
                    <button onclick="toggleSidebar()" class="pressable sidebar-toggle sidebar-toggle-mobile flex items-center justify-center w-7 h-7 rounded-sm text-zinc-500 hover:text-zinc-900 transition-colors" aria-label="Toggle sidebar">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
                    </button>
                </div>
            </div>
        </div>

        <!-- ===== Desktop layout (lg+) ===== -->
        <div class="hidden lg:flex flex-col items-stretch text-left gap-3 min-w-0 max-w-full sidebar-desktop-container">
            <div class="sidebar-desktop-header flex items-start justify-between gap-3">
                <div class="sidebar-desktop-profile flex flex-col items-start gap-3 min-w-0 flex-1">
                    <span class="${avatarFull} sidebar-desktop-avatar">${avatarImg}</span>
                    <div class="sidebar-body space-y-1 w-full sidebar-desktop-name">
                        <h1 class="text-[1.28rem] font-semibold text-zinc-950 tracking-[-0.02em] leading-tight">
                            <a href="index.html" class="hover:text-zinc-600 transition-colors">${displayName}</a>
                        </h1>
                        <p class="text-[12px] font-medium text-zinc-600 leading-snug">Ph.D. Candidate @ XJTU</p>
                    </div>
                </div>
                <button onclick="toggleSidebar()" class="pressable sidebar-toggle sidebar-toggle-desktop flex items-center justify-center w-7 h-7 rounded-sm text-zinc-500 hover:text-zinc-900 transition-colors shrink-0" aria-label="Toggle sidebar">
                    <svg class="w-4 h-4 sidebar-toggle-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
                </button>
            </div>

            <div class="sidebar-body flex flex-col gap-3">
                <div class="flex flex-col gap-1 text-[12px] w-full">
                    <a href="mailto:${personal.email}" class="link-minimal group flex items-center gap-1.5 transition-colors hover:text-emerald-600">
                        <div class="w-5 h-5 flex items-center justify-center shrink-0 transition-colors">
                            <svg class="w-3.5 h-3.5 text-zinc-500 group-hover:text-emerald-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                        </div>
                        <span class="pb-0.5">${personal.email}</span>
                    </a>
                    <a href="tel:${personal.phone.replace(/[()\s-]/g, '')}" class="link-minimal group flex items-center gap-1.5 transition-colors hover:text-emerald-600">
                        <div class="w-5 h-5 flex items-center justify-center shrink-0 transition-colors">
                            <svg class="w-3.5 h-3.5 text-zinc-500 group-hover:text-emerald-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
                        </div>
                        <span class="pb-0.5">${personal.phone}</span>
                    </a>
                    ${socialRows}
                </div>

                ${educationHtml ? `
                    <div class="sidebar-education">
                        <p class="sidebar-section-heading">Education</p>
                        ${educationHtml}
                    </div>
                ` : ''}

            </div>
        </div>
    `;

    // Restore collapsed state on load
    setSidebarCollapsed(localStorage.getItem('sidebarCollapsed') === 'true', false);
}

function setSidebarCollapsed(isCollapsed, persist = true) {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed', isCollapsed);
    document.documentElement.classList.toggle('sidebar-is-collapsed', isCollapsed);
    sidebar.querySelectorAll('.sidebar-toggle').forEach((button) => {
        button.setAttribute('aria-expanded', String(!isCollapsed));
        button.setAttribute('aria-label', isCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
    });
    if (persist) {
        localStorage.setItem('sidebarCollapsed', String(isCollapsed));
    }
}

// Toggle sidebar collapse/expand
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    setSidebarCollapsed(!sidebar.classList.contains('collapsed'));
}

// 渲染首屏概览
function renderIntro() {
    const intro = document.getElementById('intro');
    if (!intro) return;

    intro.innerHTML = `
        <div class="hero-panel reveal">
            <p class="hero-subtitle">
                I am a first-year Ph.D. student at the Faculty of Electronic and Information Engineering, Xi’an Jiaotong University, under the supervision of <a href="https://gr.xjtu.edu.cn/zhanglling/" target="_blank" rel="noopener noreferrer">Assoc. Prof. Lingling Zhang</a> and <a href="https://liujun-xjtu.github.io/zh/" target="_blank" rel="noopener noreferrer">Prof. Jun Liu</a>. I also work closely with <a href="https://www.stevens.edu/profile/yyao" target="_blank" rel="noopener noreferrer">Prof. Yudong Yao</a> at Stevens Institute of Technology. My research interests lie in vision-language models, world models, and agents.
            </p>
        </div>
    `;
}

// 渲染 Profile 部分
function renderProfile() {
    const profile = document.getElementById('profile');
    if (!profile) return;

    const { profile: profileData } = siteConfig;

    profile.innerHTML = `
        <div class="space-y-3 sm:space-y-4 reveal reveal-delay-1">
            ${profileData.education && profileData.education.length > 0 ? `
                <div class="grid gap-3 sm:gap-3.5">
                    ${profileData.education.map((edu) => {
                        const title = edu.title || `${edu.degree} in ${edu.major}`;
                        const institution = edu.institution || edu.university || '';
                        const altLogo = institution || title;
                        return `
                        <div class="card flex flex-col sm:flex-row gap-2.5 sm:gap-4 items-start sm:items-stretch">
                                ${edu.logo ? `
                                    <div class="flex w-[4.25rem] sm:w-[4.75rem] shrink-0 items-center justify-center self-stretch" aria-hidden="true">
                                        <img src="${edu.logo}" alt="${altLogo}" width="96" height="96" loading="lazy" decoding="async"
                                             class="max-h-[4.25rem] sm:max-h-[4.75rem] w-full object-contain" />
                                    </div>
                                ` : ''}
                                <div class="min-w-0 flex-1 flex flex-col gap-1 w-full">
                                    <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-1.5 sm:gap-3">
                                        <h4 class="text-zinc-900 font-semibold text-[15px] sm:text-base tracking-tight leading-snug min-w-0 flex-1">${title}</h4>
                                        <span class="text-[10px] font-semibold text-zinc-600 tabular-nums shrink-0 whitespace-nowrap px-2 py-0.5 rounded-sm bg-white/70 border border-white/80 self-start">${edu.period}</span>
                                    </div>
                                    <p class="text-zinc-600 text-sm font-medium break-words">${institution}</p>
                                    ${edu.unit ? `<p class="text-zinc-500 text-xs sm:text-sm break-words">${edu.unit}</p>` : ''}
                                    ${edu.location ? `<p class="text-zinc-600 text-[11px] sm:text-xs flex items-start gap-1.5 min-w-0">
                                        <svg class="w-3.5 h-3.5 shrink-0 text-zinc-500 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                                        <span class="min-w-0 break-words">${edu.location}</span>
                                    </p>` : ''}
                                </div>
                        </div>
                    `;
                    }).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// 渲染 Publications 区块
function renderResearch() {
    const researchContainer = document.getElementById('research');
    if (!researchContainer) return;

    const { projects } = siteConfig;
    const research = projects.filter((p) => (p.type || '').toUpperCase() === 'RESEARCH');

    const formatResearchAuthorLine = (authors) => {
        if (!authors || !Array.isArray(authors)) return "";
        return authors
            .map((a) => {
                const name = typeof a === "string" ? a : a.name;
                let marks = "";
                if (a.coFirst) {
                    marks += '<sup class="text-emerald-700 font-medium ml-px">†</sup>';
                }
                if (a.corresponding) {
                    marks += '<sup class="text-emerald-700 font-medium ml-px">*</sup>';
                }
                return name + marks;
            })
            .join(", ");
    };

    const projectLinks = (project) =>
        (project.github || project.paper)
            ? `
                        <span class="publication-links">
                            ${project.github ? `
                                <a href="${project.github}" target="_blank" rel="noopener noreferrer">GitHub</a>
                            ` : ''}
                            ${project.paper ? `
                                <a href="${project.paper}" target="_blank" rel="noopener noreferrer">Paper</a>
                            ` : ''}
                        </span>
                    `
            : '';

    const researchRows = research
        .map((project) => {
            const isPaperStyle = project.paperTitle && project.authors;
            if (isPaperStyle) {
                return `
                    <li class="publication-item">
                        <p class="publication-title">${project.paperTitle} ${projectLinks(project)}</p>
                        <p class="publication-authors">${formatResearchAuthorLine(project.authors)}</p>
                    </li>
                `;
            }
            return `
                <li class="publication-item">
                    <p class="publication-title">${project.name} ${projectLinks(project)}</p>
                    ${project.subtitle ? `<p class="publication-authors">${project.subtitle}</p>` : ''}
                    <p class="publication-note">${project.description}</p>
                </li>
            `;
        })
        .join('');

    researchContainer.innerHTML = research.length
        ? `<ul class="publication-list reveal reveal-delay-1">${researchRows}</ul>`
        : '';
}

// 渲染 Footer 部分
function renderFooter() {
    const footer = document.getElementById('footer');
    if (!footer) return;

    const { footer: footerData } = siteConfig;

    footer.innerHTML = `
        <div class="pt-3 sm:pt-4 flex flex-col items-center gap-1 text-zinc-500 border-t border-zinc-200">
            <div class="flex flex-col items-center gap-0.5">
                <p class="text-[11px] font-medium tracking-[0.12em] uppercase text-zinc-500">${footerData.systemId}</p>
                ${footerData.designPattern ? `<p class="text-[10px] text-zinc-600">${footerData.designPattern}</p>` : ''}
            </div>
        </div>
    `;
}

/**
 * IntersectionObserver — 元素进入视口时添加 .revealed 触发动画
 */
function observeReveal() {
    const els = document.querySelectorAll('.reveal');
    if (els.length) {
        const obs = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('revealed');
                        obs.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.08, rootMargin: '0px 0px -20px 0px' }
        );
        els.forEach((el) => obs.observe(el));
    }
}

document.addEventListener('DOMContentLoaded', function() {
    renderThemeDock();
    renderHeader();
    applyTheme(document.documentElement.dataset.theme || getPreferredTheme());

    renderIntro();
    if (document.getElementById('research')) {
        renderResearch();
    }
    renderFooter();
    observeReveal();
});

const themeMediaQuery = window.matchMedia?.('(prefers-color-scheme: dark)');
const handleSystemThemeChange = (event) => {
    try {
        if (localStorage.getItem('theme')) return;
    } catch (error) {}
    applyTheme(event.matches ? 'dark' : 'light');
};
if (themeMediaQuery?.addEventListener) {
    themeMediaQuery.addEventListener('change', handleSystemThemeChange);
} else if (themeMediaQuery?.addListener) {
    themeMediaQuery.addListener(handleSystemThemeChange);
}
