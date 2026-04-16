// 渲染左侧边栏：头像、基本信息、外链与快速导航（首页 / 博客共用布局）
function renderHeader() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    const { personal } = siteConfig;
    const displayName = personal.name.replace(/_/g, ' ');
    const path =
        typeof window !== 'undefined' && window.location.pathname ? window.location.pathname : '';
    const onBlogPage = /blog\.html$/i.test(path);

    const avatarFrame =
        'relative block shrink-0 overflow-hidden rounded-lg border border-zinc-200 bg-zinc-100 shadow-sm shadow-zinc-200/60 w-[7.5rem] sm:w-32 aspect-[295/413] mx-auto lg:mx-0';
    const avatarBlock = personal.photo
        ? `<span class="${avatarFrame}">
                <img src="${personal.photo}" alt="${displayName}" width="295" height="413" decoding="async" class="absolute inset-0 h-full w-full object-cover object-top" onerror="this.classList.add('hidden'); this.nextElementSibling.classList.remove('hidden')" />
                <span class="hidden absolute inset-0 bg-zinc-200/80 flex items-center justify-center text-zinc-600 font-semibold text-sm tracking-tight" aria-hidden="true">XP</span>
           </span>`
        : `<div class="${avatarFrame} flex items-center justify-center text-zinc-600 font-semibold text-sm tracking-tight">XP</div>`;

    const sectionAnchors = [
        { id: 'section-about', label: 'About' },
        { id: 'section-research-interests', label: 'Research interests' },
        { id: 'section-research', label: 'Research' },
        { id: 'section-projects', label: 'Projects' },
        { id: 'section-achievements', label: 'Achievements' },
    ];

    const navLinks = onBlogPage
        ? [
              { href: '#section-blog', label: 'Blog' },
              ...sectionAnchors.map((s) => ({ href: `index.html#${s.id}`, label: s.label })),
          ]
        : sectionAnchors.map((s) => ({ href: `#${s.id}`, label: s.label }));

    const quickNavTitle = onBlogPage ? 'Navigate' : 'On this page';

    const quickNavHtml = `
            <nav class="hidden lg:block w-full min-w-0 max-w-full pt-4 mt-1 border-t border-zinc-200 overflow-x-hidden" aria-label="Quick navigation">
                <p class="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.12em] mb-2 text-center lg:text-left">${quickNavTitle}</p>
                <ul class="flex flex-col gap-0.5 text-sm min-w-0">
                    ${navLinks
                        .map(
                            (l) =>
                                `<li class="min-w-0"><a href="${l.href}" class="text-zinc-600 hover:text-emerald-700 py-1.5 px-1 rounded-md hover:bg-zinc-50 transition-colors block text-center lg:text-left break-words">${l.label}</a></li>`
                        )
                        .join('')}
                </ul>
            </nav>
        `;

    sidebar.innerHTML = `
        <div class="flex flex-col items-center lg:items-stretch text-center lg:text-left gap-3 sm:gap-4 lg:gap-5 fade-in staggered-1 min-w-0 max-w-full overflow-x-hidden">
            ${avatarBlock}
            <div class="flex flex-col gap-2 sm:gap-2.5 lg:gap-3.5 min-w-0 w-full max-w-full overflow-x-hidden">
                <h1 class="text-lg sm:text-xl lg:text-2xl font-semibold text-zinc-900 tracking-[-0.02em] leading-tight">
                    <a href="index.html" class="hover:text-zinc-600 transition-colors">${displayName}</a>
                </h1>
                <p class="text-[9px] sm:text-[10px] lg:text-[11px] font-medium tracking-[0.11em] uppercase text-zinc-500 leading-snug">
                    ${personal.bootMessage}
                </p>
                <div class="flex flex-col gap-1.5 sm:gap-2 lg:gap-2.5 text-xs sm:text-sm lg:text-sm w-full pt-0.5 lg:pt-1">
                    <a href="mailto:${personal.email}" class="link-minimal min-h-[44px] sm:min-h-0 inline-flex items-center justify-center sm:justify-start gap-2 sm:gap-2.5 break-all transition-colors hover:text-emerald-600">
                        <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4 shrink-0 opacity-75 hover:opacity-100" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
                        <span class="text-left">${personal.email}</span>
                    </a>
                    <a href="tel:${personal.phone.replace(/[()\s-]/g, '')}" class="link-minimal min-h-[44px] sm:min-h-0 inline-flex items-center justify-center sm:justify-start gap-2 sm:gap-2.5 whitespace-nowrap transition-colors hover:text-emerald-600">
                        <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4 shrink-0 opacity-75 hover:opacity-100" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
                        ${personal.phone}
                    </a>
                </div>

                <nav class="grid grid-cols-2 sm:flex sm:flex-wrap gap-1.5 sm:gap-2 lg:gap-2.5 justify-center sm:justify-center lg:justify-start pt-1 lg:pt-1.5 min-w-0 max-w-full overflow-x-hidden" aria-label="${onBlogPage ? 'Site navigation' : 'Social links'}">
                    ${onBlogPage ? `
                    <a href="index.html" class="pill col-span-2 sm:col-span-auto inline-flex items-center justify-center gap-2 no-underline px-3 sm:px-3.5 lg:px-4 py-2 sm:py-2.5 lg:py-2.5 text-xs sm:text-[13px] lg:text-sm border-emerald-200 bg-emerald-50 text-emerald-800 hover:border-emerald-300">
                        <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
                        Home
                    </a>
                    ` : `
                    <a href="blog.html" class="pill col-span-2 sm:col-span-auto inline-flex items-center justify-center gap-2 no-underline px-3 sm:px-3.5 lg:px-4 py-2 sm:py-2.5 lg:py-2.5 text-xs sm:text-[13px] lg:text-sm">
                        <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/></svg>
                        Blog
                    </a>
                    `}
                    ${personal.social.github ? `
                        <a href="${personal.social.github}" target="_blank" rel="noopener noreferrer" class="pill inline-flex items-center justify-center gap-1.5 sm:gap-2 lg:gap-2 no-underline px-3 sm:px-3.5 lg:px-4 py-2 sm:py-2.5 lg:py-2.5 text-xs sm:text-[13px] lg:text-sm" title="GitHub" aria-label="GitHub profile">
                            <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4 shrink-0" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                            <span class="hidden sm:inline">GitHub</span>
                        </a>
                    ` : ''}
                    ${personal.social.googleScholar ? `
                        <a href="${personal.social.googleScholar}" target="_blank" rel="noopener noreferrer" class="pill inline-flex items-center justify-center gap-1.5 sm:gap-2 lg:gap-2 no-underline px-3 sm:px-3.5 lg:px-4 py-2 sm:py-2.5 lg:py-2.5 text-xs sm:text-[13px] lg:text-sm" title="Google Scholar" aria-label="Google Scholar profile">
                            <svg class="w-3.5 h-3.5 lg:w-4 lg:h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l9-5-9-5-9 5 9 5z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/></svg>
                            <span class="hidden sm:inline">Scholar</span>
                        </a>
                    ` : ''}
                    ${personal.social.huggingface ? `
                        <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" class="pill inline-flex items-center justify-center gap-1.5 sm:gap-2 lg:gap-2 no-underline px-3 sm:px-3.5 lg:px-4 py-2 sm:py-2.5 lg:py-2.5 text-xs sm:text-[13px] lg:text-sm">
                            <span class="text-sm lg:text-base">🤗</span> <span class="hidden sm:inline">HF</span>
                        </a>
                    ` : ''}
                </nav>

                ${quickNavHtml}
            </div>
        </div>
    `;
}

// 渲染 Profile 部分
function renderProfile() {
    const profile = document.getElementById('profile');
    if (!profile) return;

    const { profile: profileData } = siteConfig;

    profile.innerHTML = `
        <div class="space-y-3 sm:space-y-4 fade-in staggered-2">
            ${profileData.education && profileData.education.length > 0 ? `
                <div class="grid gap-3 sm:gap-3.5">
                    ${profileData.education.map((edu) => {
                        const title = edu.title || `${edu.degree} in ${edu.major}`;
                        const institution = edu.institution || edu.university || '';
                        const altLogo = institution || title;
                        return `
                        <div class="flex gap-2.5 sm:gap-3.5 items-stretch">
                                ${edu.logo ? `
                                    <div class="flex w-[4.25rem] sm:w-[4.75rem] shrink-0 items-center justify-center self-stretch" aria-hidden="true">
                                        <img src="${edu.logo}" alt="${altLogo}" width="96" height="96" loading="lazy" decoding="async"
                                             class="max-h-full w-full h-full object-contain" />
                                    </div>
                                ` : ''}
                                <div class="min-w-0 flex-1 flex flex-col gap-1">
                                    <div class="flex items-start justify-between gap-2 sm:gap-3">
                                        <h4 class="text-zinc-900 font-semibold text-[15px] sm:text-base tracking-tight leading-snug min-w-0 flex-1">${title}</h4>
                                        <span class="text-[10px] font-medium text-zinc-600 tabular-nums shrink-0 whitespace-nowrap px-2.5 py-1 rounded-full bg-zinc-100 border border-zinc-200">${edu.period}</span>
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

// 渲染研究兴趣（原 domains，无 Programming / Frameworks）
function renderResearchInterests() {
    const el = document.getElementById('research-interests');
    if (!el) return;

    const { skills } = siteConfig;
    const domains = skills.domains || [];

    el.innerHTML = `
        <div class="flex flex-wrap gap-1.5 fade-in staggered-3">
            ${domains.map((domain) => `<span class="text-xs font-medium px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 cursor-default">${domain}</span>`).join('')}
        </div>
    `;
}

// 渲染 Research（#research）与 Projects（#projects）两个区块
function renderProjects() {
    const researchContainer = document.getElementById('research');
    const projectsContainer = document.getElementById('projects');
    if (!researchContainer && !projectsContainer) return;

    const { projects } = siteConfig;
    const research = projects.filter((p) => (p.type || '').toUpperCase() === 'RESEARCH');
    const projectItems = projects.filter((p) => (p.type || '').toUpperCase() !== 'RESEARCH');

    const techSpans = (project) =>
        project.technologies
            .map(
                (tech) =>
                    `<span class="text-[10px] font-medium text-zinc-600 px-2 py-0.5 rounded-md bg-zinc-100 border border-zinc-200">${tech}</span>`
            )
            .join('');

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
                        <div class="flex flex-wrap gap-3 sm:gap-4 shrink-0">
                            ${project.github ? `
                                <a href="${project.github}" target="_blank" rel="noopener noreferrer" class="text-xs font-semibold text-zinc-500 hover:text-accent-color transition-colors inline-flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                                    GitHub
                                </a>
                            ` : ''}
                            ${project.paper ? `
                                <a href="${project.paper}" target="_blank" rel="noopener noreferrer" class="text-xs font-semibold text-zinc-500 hover:text-accent-color transition-colors inline-flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
                                    Paper
                                </a>
                            ` : ''}
                        </div>
                    `
            : '';

    const researchRows = research
        .map((project) => {
            const isPaperStyle = project.paperTitle && project.authors;
            if (isPaperStyle) {
                return `
                <article class="rounded-lg border border-zinc-200 bg-zinc-50 px-3.5 py-3 sm:px-4 sm:py-3.5 min-w-0 transition-colors sm:hover:border-emerald-300/60 sm:hover:bg-zinc-100/80">
                    <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
                        <div class="min-w-0 flex-1 space-y-1">
                            <h4 class="text-[15px] sm:text-base font-semibold text-zinc-900 tracking-tight leading-snug">${project.paperTitle}</h4>
                            <p class="text-sm text-zinc-600 leading-relaxed pt-1">${formatResearchAuthorLine(project.authors)}</p>
                        </div>
                        <div class="shrink-0 sm:pt-0.5">${projectLinks(project)}</div>
                    </div>
                </article>
            `;
            }
            return `
                <article class="rounded-lg border border-zinc-200 bg-zinc-50 px-3.5 py-3 sm:px-4 sm:py-3.5 min-w-0 transition-colors sm:hover:border-emerald-300/60 sm:hover:bg-zinc-100/80">
                    <div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between lg:gap-6">
                        <div class="min-w-0 flex-1 space-y-1">
                            <h4 class="text-[15px] sm:text-base font-semibold text-zinc-900 tracking-tight break-words">${project.name}</h4>
                            ${project.subtitle ? `<p class="text-sm font-medium text-zinc-600">${project.subtitle}</p>` : ''}
                            <p class="text-sm text-zinc-600 leading-relaxed">${project.description}</p>
                        </div>
                        <div class="flex flex-col gap-2.5 sm:flex-row sm:flex-wrap sm:items-center lg:flex-col lg:items-end xl:flex-row xl:items-center shrink-0 lg:pt-0.5">
                            <div class="flex flex-wrap gap-1.5">${techSpans(project)}</div>
                            ${projectLinks(project)}
                        </div>
                    </div>
                </article>
            `;
        })
        .join('');

    const projectCards = projectItems
        .map(
            (project) => `
                <article class="card group relative overflow-hidden flex flex-col gap-2.5 sm:gap-3 min-w-0">
                    <div class="hidden lg:block absolute top-0 right-0 p-3 opacity-0 group-hover:opacity-100 transition-all transform translate-x-2 group-hover:translate-x-0 pointer-events-none">
                        <svg class="w-5 h-5 text-accent-color" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                        </svg>
                    </div>
                    
                    <div class="flex flex-col gap-1">
                        <span class="text-[9px] font-semibold uppercase tracking-[0.2em] text-accent-color">${project.type}</span>
                        <h4 class="text-base sm:text-lg lg:text-[1.125rem] xl:text-xl font-semibold text-zinc-900 tracking-tight break-words">${project.name}</h4>
                        ${project.subtitle ? `<p class="text-sm font-medium text-zinc-600">${project.subtitle}</p>` : ''}
                    </div>

                    <p class="text-sm text-zinc-600 leading-relaxed">
                        ${project.description}
                    </p>

                    <div class="flex flex-wrap gap-1.5">
                        ${techSpans(project)}
                    </div>

                    ${(project.github || project.paper) ? `
                        <div class="flex flex-wrap gap-4 pt-2.5 border-t border-zinc-200">
                            ${project.github ? `
                                <a href="${project.github}" target="_blank" rel="noopener noreferrer" class="text-xs font-semibold text-zinc-500 hover:text-accent-color transition-colors inline-flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                                    GitHub
                                </a>
                            ` : ''}
                            ${project.paper ? `
                                <a href="${project.paper}" target="_blank" rel="noopener noreferrer" class="text-xs font-semibold text-zinc-500 hover:text-accent-color transition-colors inline-flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
                                    Paper
                                </a>
                            ` : ''}
                        </div>
                    ` : ''}
                </article>
            `
        )
        .join('');

    if (researchContainer) {
        researchContainer.innerHTML = research.length
            ? `<div class="flex flex-col gap-2 sm:gap-2.5 fade-in staggered-4">${researchRows}</div>`
            : '';
    }
    if (projectsContainer) {
        projectsContainer.innerHTML = projectItems.length
            ? `<div class="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 lg:gap-5 fade-in staggered-4">${projectCards}</div>`
            : '';
    }
}

// 渲染 Achievements 部分
function renderAchievements() {
    const achievementsContainer = document.getElementById('achievements');
    if (!achievementsContainer) return;

    const { achievements } = siteConfig;

    achievementsContainer.innerHTML = `
        <div class="grid gap-2 sm:gap-2.5 fade-in staggered-5">
            ${achievements.map(achievement => `
                <div class="flex flex-row items-start justify-between gap-2 sm:gap-3 p-3 sm:p-3.5 rounded-md sm:rounded-lg bg-zinc-50 border border-zinc-200 max-sm:active:bg-zinc-100 sm:hover:border-emerald-300/50 sm:hover:bg-zinc-100/80 transition-all min-w-0">
                    <div class="flex flex-col gap-0 min-w-0 flex-1 pr-1">
                        <span class="text-[13px] sm:text-sm font-semibold text-zinc-900 break-words">${achievement.title}</span>
                        ${achievement.subtitle ? `<span class="text-xs font-medium text-zinc-600">${achievement.subtitle}</span>` : ''}
                    </div>
                    <span class="text-[11px] font-medium tabular-nums text-zinc-600 shrink-0 whitespace-nowrap text-right self-start px-2.5 py-1 rounded-md bg-zinc-100 border border-zinc-200">${achievement.date}</span>
                </div>
            `).join('')}
        </div>
    `;
}

// 渲染 Footer 部分
function renderFooter() {
    const footer = document.getElementById('footer');
    if (!footer) return;

    const { footer: footerData } = siteConfig;

    footer.innerHTML = `
        <div class="pt-3 sm:pt-4 flex flex-col items-center gap-2 text-zinc-500">
            <div class="flex gap-2 items-center">
                <span class="w-1.5 h-1.5 rounded-full bg-accent-color shadow-[0_0_12px_rgba(16,163,127,0.5)]"></span>
                <span class="w-1 h-1 rounded-full bg-zinc-400"></span>
                <span class="w-1 h-1 rounded-full bg-zinc-300"></span>
            </div>
            <div class="flex flex-col items-center gap-1">
                <p class="text-[11px] font-medium tracking-[0.12em] uppercase text-zinc-500">${footerData.systemId}</p>
                ${footerData.designPattern ? `<p class="text-[10px] text-zinc-600">${footerData.designPattern}</p>` : ''}
            </div>
        </div>
    `;
}

document.addEventListener('DOMContentLoaded', function() {
    renderHeader();

    if (document.getElementById('profile')) {
        renderProfile();
    }
    if (document.getElementById('research-interests')) {
        renderResearchInterests();
    }
    if (document.getElementById('research') || document.getElementById('projects')) {
        renderProjects();
    }
    if (document.getElementById('achievements')) {
        renderAchievements();
    }

    renderFooter();
});
