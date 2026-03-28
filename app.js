// 渲染 Header 部分
function renderHeader() {
    const header = document.getElementById('header');
    if (!header) return;

    const { personal } = siteConfig;
    const displayName = personal.name.replace(/_/g, ' ');
    const path = (typeof window !== 'undefined' && window.location.pathname) ? window.location.pathname : '';
    const onBlogPage = /blog\.html$/i.test(path);
    /* 一寸证件照比例 295×413（宽:高），网页上约 100–118px 宽，接近表格/简历贴照尺寸 */
    const avatarFrame =
        'relative block shrink-0 overflow-hidden rounded-md border border-white/[0.14] bg-zinc-900/50 shadow-sm shadow-black/40 w-[92px] sm:w-[100px] md:w-[110px] lg:w-[118px] aspect-[295/413] mx-auto sm:mx-0';
    const avatarBlock = personal.photo
        ? `<span class="${avatarFrame}">
                <img src="${personal.photo}" alt="${displayName}" width="295" height="413" decoding="async" class="absolute inset-0 h-full w-full object-cover object-top" onerror="this.classList.add('hidden'); this.nextElementSibling.classList.remove('hidden')" />
                <span class="hidden absolute inset-0 bg-white/[0.06] flex items-center justify-center text-white font-semibold text-sm tracking-tight" aria-hidden="true">XP</span>
           </span>`
        : `<div class="${avatarFrame} flex items-center justify-center text-white font-semibold text-sm tracking-tight">XP</div>`;

    header.innerHTML = `
        <div class="flex flex-col sm:flex-row gap-5 sm:gap-6 md:gap-8 lg:gap-10 items-center sm:items-start text-center sm:text-left fade-in staggered-1">
            ${avatarBlock}
            <div class="flex flex-col gap-2.5 sm:gap-3 md:gap-4 min-w-0 flex-1 w-full sm:w-auto pt-0 sm:pt-1">
                <p class="text-[10px] sm:text-[11px] font-medium tracking-[0.12em] uppercase text-zinc-500">
                    ${personal.bootMessage}
                </p>
                <h1 class="text-[1.625rem] sm:text-3xl md:text-[2.25rem] lg:text-[2.5rem] font-semibold text-white tracking-[-0.03em] leading-[1.15] sm:leading-tight">
                    <a href="index.html" class="hover:text-white/80 transition-colors">${displayName}</a>
                </h1>
                <p class="text-[14px] sm:text-[15px] md:text-base text-zinc-400 leading-snug max-w-2xl font-normal mx-auto sm:mx-0 break-words">
                    ${personal.title.replace(/_/g, ' ').replace('[', ' — ').replace(']', '')}
                </p>
                <div class="flex flex-col sm:flex-row sm:flex-wrap justify-center sm:justify-start gap-x-6 gap-y-2 text-sm">
                <a href="mailto:${personal.email}" class="link-minimal min-h-[44px] sm:min-h-0 inline-flex items-center justify-center sm:justify-start">${personal.email}</a>
                <a href="tel:${personal.phone.replace(/[()\s-]/g, '')}" class="link-minimal min-h-[44px] sm:min-h-0 inline-flex items-center justify-center sm:justify-start">${personal.phone}</a>
            </div>

            <nav class="flex flex-wrap gap-2 sm:gap-1.5 pt-1 justify-center sm:justify-start" aria-label="${onBlogPage ? 'Site navigation' : 'Social links'}">
                ${onBlogPage ? `
                <a href="index.html" class="pill inline-flex items-center justify-center gap-2 no-underline max-sm:min-h-[44px] max-sm:px-3.5 max-sm:py-2.5 border-emerald-500/25 bg-emerald-500/5 text-emerald-400/95 hover:border-emerald-500/40">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
                    Home
                </a>
                ` : `
                <a href="blog.html" class="pill inline-flex items-center justify-center gap-2 no-underline max-sm:min-h-[44px] max-sm:px-3.5 max-sm:py-2.5">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/></svg>
                    Blog
                </a>
                `}
                ${personal.social.googleScholar ? `
                    <a href="${personal.social.googleScholar}" target="_blank" rel="noopener noreferrer" class="pill inline-flex items-center justify-center gap-2 no-underline max-sm:min-h-[44px] max-sm:px-3.5 max-sm:py-2.5" title="Google Scholar" aria-label="Google Scholar profile">
                        <svg class="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
                        <span class="whitespace-nowrap">Google Scholar</span>
                    </a>
                ` : ''}
                ${personal.social.huggingface ? `
                    <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" class="pill inline-flex items-center justify-center gap-2 no-underline max-sm:min-h-[44px] max-sm:px-3.5 max-sm:py-2.5">
                        <span class="text-xs">🤗</span> Hugging Face
                    </a>
                ` : ''}
            </nav>
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
        <div class="space-y-5 sm:space-y-6 fade-in staggered-2">
            ${profileData.education && profileData.education.length > 0 ? `
                <div class="grid gap-4 sm:gap-5">
                    ${profileData.education.map((edu) => {
                        const title = edu.title || `${edu.degree} in ${edu.major}`;
                        const institution = edu.institution || edu.university || '';
                        const altLogo = institution || title;
                        return `
                        <div class="flex gap-2.5 sm:gap-3.5 items-start">
                                ${edu.logo ? `
                                    <img src="${edu.logo}" alt="${altLogo}" width="48" height="48" loading="lazy" decoding="async"
                                         class="w-10 h-10 sm:w-11 sm:h-12 shrink-0 object-contain rounded-lg bg-white/[0.06] p-1 sm:p-1.5 border border-white/[0.1] mt-0.5" />
                                ` : ''}
                                <div class="min-w-0 flex-1 flex flex-col gap-1">
                                    <div class="flex items-start justify-between gap-2 sm:gap-3">
                                        <h4 class="text-white font-semibold text-[15px] sm:text-base tracking-tight leading-snug min-w-0 flex-1">${title}</h4>
                                        <span class="text-[10px] font-medium text-zinc-500 tabular-nums shrink-0 whitespace-nowrap px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08]">${edu.period}</span>
                                    </div>
                                    <p class="text-zinc-400 text-sm font-medium break-words">${institution}</p>
                                    ${edu.unit ? `<p class="text-zinc-500 text-xs sm:text-sm break-words">${edu.unit}</p>` : ''}
                                    ${edu.location ? `<p class="text-zinc-600 text-[11px] sm:text-xs">${edu.location}</p>` : ''}
                                </div>
                        </div>
                    `;
                    }).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// 渲染 Skills 部分
function renderSkills() {
    const skillsContainer = document.getElementById('skills');
    if (!skillsContainer) return;

    const { skills } = siteConfig;

    skillsContainer.innerHTML = `
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8 fade-in staggered-3">
            <div class="flex flex-col gap-2 min-w-0">
                <h4 class="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Programming</h4>
                <div class="flex flex-wrap gap-1.5">
                    ${skills.programming.map(skill => `<span class="pill cursor-default">${skill}</span>`).join('')}
                </div>
            </div>
            <div class="flex flex-col gap-2 min-w-0">
                <h4 class="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Frameworks</h4>
                <div class="flex flex-wrap gap-1.5">
                    ${skills.frameworks.map(skill => `<span class="pill cursor-default">${skill}</span>`).join('')}
                </div>
            </div>
            <div class="flex flex-col gap-2 pt-1 sm:pt-0 sm:col-span-2 lg:col-span-1 min-w-0">
                <h4 class="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Research domains</h4>
                <div class="flex flex-wrap gap-1.5">
                    ${skills.domains.map(domain => `<span class="text-xs font-medium px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400/95 border border-emerald-500/20 cursor-default">${domain}</span>`).join('')}
                </div>
            </div>
        </div>
    `;
}

// 渲染 Projects 部分
function renderProjects() {
    const projectsContainer = document.getElementById('projects');
    if (!projectsContainer) return;

    const { projects } = siteConfig;

    projectsContainer.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-5 lg:gap-6 fade-in staggered-4">
            ${projects.map((project) => `
                <article class="card group relative overflow-hidden flex flex-col gap-3 sm:gap-3.5 min-w-0">
                    <div class="hidden lg:block absolute top-0 right-0 p-3 opacity-0 group-hover:opacity-100 transition-all transform translate-x-2 group-hover:translate-x-0 pointer-events-none">
                        <svg class="w-5 h-5 text-accent-color" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                        </svg>
                    </div>
                    
                    <div class="flex flex-col gap-1">
                        <span class="text-[9px] font-semibold uppercase tracking-[0.2em] text-accent-color">${project.type}</span>
                        <h4 class="text-base sm:text-lg lg:text-[1.125rem] xl:text-xl font-semibold text-white tracking-tight break-words">${project.name}</h4>
                        ${project.subtitle ? `<p class="text-sm font-medium text-zinc-500">${project.subtitle}</p>` : ''}
                    </div>

                    <p class="text-sm text-zinc-400 leading-relaxed">
                        ${project.description}
                    </p>

                    <div class="flex flex-wrap gap-1.5">
                        ${project.technologies.map(tech => `<span class="text-[10px] font-medium text-zinc-400 px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.08]">${tech}</span>`).join('')}
                    </div>

                    ${(project.github || project.paper) ? `
                        <div class="flex flex-wrap gap-5 pt-3.5 border-t border-white/[0.08]">
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
            `).join('')}
        </div>
    `;
}

// 渲染 Achievements 部分
function renderAchievements() {
    const achievementsContainer = document.getElementById('achievements');
    if (!achievementsContainer) return;

    const { achievements } = siteConfig;

    achievementsContainer.innerHTML = `
        <div class="grid gap-2 sm:gap-2.5 fade-in staggered-5">
            ${achievements.map(achievement => `
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 p-3 sm:p-3.5 rounded-md sm:rounded-lg bg-white/[0.03] border border-white/[0.08] max-sm:active:bg-white/[0.05] sm:hover:border-emerald-500/35 sm:hover:bg-white/[0.04] transition-all">
                    <div class="flex flex-col gap-0 min-w-0">
                        <span class="text-[13px] sm:text-sm font-semibold text-white break-words">${achievement.title}</span>
                        ${achievement.subtitle ? `<span class="text-xs font-medium text-zinc-500">${achievement.subtitle}</span>` : ''}
                    </div>
                    <span class="text-[11px] font-medium tabular-nums text-zinc-500 shrink-0 px-2.5 py-1 rounded-md bg-white/[0.05] border border-white/[0.06]">${achievement.date}</span>
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
        <div class="pt-6 sm:pt-8 lg:pt-10 border-t border-white/[0.08] flex flex-col items-center gap-2.5 sm:gap-3 text-zinc-500">
            <div class="flex gap-2 items-center">
                <span class="w-1.5 h-1.5 rounded-full bg-accent-color shadow-[0_0_12px_rgba(16,163,127,0.5)]"></span>
                <span class="w-1 h-1 rounded-full bg-zinc-600"></span>
                <span class="w-1 h-1 rounded-full bg-zinc-700"></span>
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
    if (document.getElementById('skills')) {
        renderSkills();
    }
    if (document.getElementById('projects')) {
        renderProjects();
    }
    if (document.getElementById('achievements')) {
        renderAchievements();
    }

    renderFooter();
});
