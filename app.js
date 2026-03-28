// 渲染 Header 部分
function renderHeader() {
    const header = document.getElementById('header');
    if (!header) return;

    const { personal } = siteConfig;

    header.innerHTML = `
        <div class="flex flex-col gap-6">
            <p class="text-sm text-neutral-500">
                ${personal.bootMessage}
            </p>

            <h1 class="text-3xl md:text-4xl font-semibold text-neutral-900 tracking-tight">
                <a href="index.html" class="hover:text-neutral-600 transition-colors">${personal.name.replace(/_/g, ' ')}</a>
            </h1>

            <p class="text-base text-neutral-600 leading-relaxed max-w-xl">
                ${personal.title.replace(/_/g, ' ').replace('[', ' — ').replace(']', '')}
            </p>

            <div class="flex flex-col sm:flex-row sm:flex-wrap gap-x-8 gap-y-2 text-sm text-neutral-600">
                <a href="mailto:${personal.email}" class="hover:text-neutral-900 transition-colors border-b border-transparent hover:border-neutral-300">${personal.email}</a>
                <a href="tel:${personal.phone.replace(/[()\s-]/g, '')}" class="hover:text-neutral-900 transition-colors border-b border-transparent hover:border-neutral-300">${personal.phone}</a>
            </div>

            ${(personal.social && (personal.social.x || personal.social.huggingface || personal.social.reddit)) || true ? `
                <nav class="flex flex-wrap gap-4 pt-2">
                    <a href="blog.html" class="text-sm text-neutral-500 hover:text-neutral-900 transition-colors inline-flex items-center gap-2">
                        <svg class="w-4 h-4 opacity-60" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                        </svg>
                        Blog
                    </a>
                    ${personal.social.x ? `
                        <a href="${personal.social.x}" target="_blank" rel="noopener noreferrer" class="text-sm text-neutral-500 hover:text-neutral-900 transition-colors inline-flex items-center gap-2">
                            <svg class="w-4 h-4 opacity-60" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                            </svg>
                            X
                        </a>
                    ` : ''}
                    ${personal.social.huggingface ? `
                        <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" class="text-sm text-neutral-500 hover:text-neutral-900 transition-colors inline-flex items-center gap-2">
                            <svg class="w-4 h-4 opacity-60" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 2.4c5.302 0 9.6 4.298 9.6 9.6 0 5.302-4.298 9.6-9.6 9.6-5.302 0-9.6-4.298-9.6-9.6 0-5.302 4.298-9.6 9.6-9.6zm-2.4 4.8c-1.325 0-2.4 1.075-2.4 2.4s1.075 2.4 2.4 2.4 2.4-1.075 2.4-2.4-1.075-2.4-2.4-2.4zm4.8 0c-1.325 0-2.4 1.075-2.4 2.4s1.075 2.4 2.4 2.4 2.4-1.075 2.4-2.4-1.075-2.4-2.4-2.4zm-2.4 7.2c-2.65 0-4.8 1.35-4.8 3v1.2h9.6v-1.2c0-1.65-2.15-3-4.8-3z"/>
                            </svg>
                            Hugging Face
                        </a>
                    ` : ''}
                    ${personal.social.reddit ? `
                        <a href="${personal.social.reddit}" target="_blank" rel="noopener noreferrer" class="text-sm text-neutral-500 hover:text-neutral-900 transition-colors inline-flex items-center gap-2">
                            <svg class="w-4 h-4 opacity-60" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                            </svg>
                            Reddit
                        </a>
                    ` : ''}
                </nav>
            ` : ''}
        </div>
    `;
}

// 渲染 Profile 部分
function renderProfile() {
    const profile = document.getElementById('profile');
    if (!profile) return;

    const { profile: profileData } = siteConfig;

    profile.innerHTML = `
        <div class="space-y-6">
            ${profileData.education && profileData.education.length > 0 ? `
                <ul class="space-y-4">
                    ${profileData.education.map((edu) => `
                        <li class="text-neutral-700 leading-relaxed">
                            <span class="text-neutral-900 font-medium">${edu.degree}</span>
                            <span class="text-neutral-400 mx-2">·</span>
                            <span>${edu.major}</span>
                            <span class="text-neutral-400 mx-2">·</span>
                            <span>${edu.university}</span>
                            ${edu.period ? `<span class="block sm:inline sm:ml-2 text-sm text-neutral-500 mt-1 sm:mt-0">${edu.period}</span>` : ''}
                        </li>
                    `).join('')}
                </ul>
            ` : ''}
            <p class="text-neutral-600 leading-relaxed text-sm md:text-base">
                <span class="text-neutral-400">Focus.</span> ${profileData.focus}
            </p>
        </div>
    `;
}

// 渲染 Skills 部分
function renderSkills() {
    const skillsContainer = document.getElementById('skills');
    if (!skillsContainer) return;

    const { skills } = siteConfig;

    skillsContainer.innerHTML = `
        <div class="space-y-8">
            <div>
                <h4 class="text-sm font-medium text-neutral-900 mb-3">Programming</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.programming.map(skill => `
                        <span class="text-sm text-neutral-600 px-0 py-0.5">${skill}</span>
                    `).join('<span class="text-neutral-300">·</span>')}
                </div>
            </div>
            <div>
                <h4 class="text-sm font-medium text-neutral-900 mb-3">Frameworks / libraries</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.frameworks.map(skill => `
                        <span class="text-sm text-neutral-600 px-0 py-0.5">${skill}</span>
                    `).join('<span class="text-neutral-300">·</span>')}
                </div>
            </div>
            <div>
                <h4 class="text-sm font-medium text-neutral-900 mb-3">Research domains</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.domains.map(domain => `
                        <span class="text-sm text-neutral-600 px-0 py-0.5">${domain}</span>
                    `).join('<span class="text-neutral-300">·</span>')}
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
        <div class="space-y-12">
            ${projects.map((project) => `
                <article class="group">
                    <div class="flex flex-col gap-1 mb-3">
                        <span class="text-[11px] uppercase tracking-wider text-neutral-400">${project.type === 'RESEARCH' ? 'Research' : 'Project'}</span>
                        <h4 class="text-lg font-medium text-neutral-900">${project.name}</h4>
                        ${project.subtitle ? `<p class="text-sm text-neutral-500">${project.subtitle}</p>` : ''}
                    </div>
                    <p class="text-sm text-neutral-600 leading-relaxed mb-4">
                        ${project.description}
                    </p>
                    <div class="flex flex-wrap gap-2 mb-4">
                        ${project.technologies.map(tech => `
                            <span class="text-xs text-neutral-500">${tech}</span>
                        `).join('<span class="text-neutral-300 text-xs">·</span>')}
                    </div>
                    ${(project.github || project.paper) ? `
                        <div class="flex flex-wrap gap-6 text-sm">
                            ${project.github ? `
                                <a href="${project.github}" target="_blank" rel="noopener noreferrer" class="text-neutral-500 hover:text-neutral-900 border-b border-neutral-200 hover:border-neutral-400 transition-colors inline-flex items-center gap-1.5">
                                    <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                                    </svg>
                                    GitHub
                                </a>
                            ` : ''}
                            ${project.paper ? `
                                <a href="${project.paper}" target="_blank" rel="noopener noreferrer" class="text-neutral-500 hover:text-neutral-900 border-b border-neutral-200 hover:border-neutral-400 transition-colors inline-flex items-center gap-1.5">
                                    <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                                        <path d="M14,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V8L14,3M19,19H5V5H13V9H19M17,17H7V15H17M17,13H7V11H17M17,9H7V7H17V9Z"/>
                                    </svg>
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
        <ul class="space-y-0 divide-y divide-neutral-200 border-t border-b border-neutral-200">
            ${achievements.map(achievement => `
                <li class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1 py-4">
                    <div>
                        <span class="text-neutral-900 font-medium">${achievement.title}</span>
                        ${achievement.subtitle ? `<span class="block text-sm text-neutral-500 mt-0.5">${achievement.subtitle}</span>` : ''}
                    </div>
                    <span class="text-xs text-neutral-400 sm:shrink-0 sm:ml-4">${achievement.date}</span>
                </li>
            `).join('')}
        </ul>
    `;
}

// 渲染 Footer 部分
function renderFooter() {
    const footer = document.getElementById('footer');
    if (!footer) return;

    const { footer: footerData } = siteConfig;

    footer.innerHTML = `
        <p>${footerData.systemId}</p>
        ${footerData.designPattern ? `<p class="mt-1">${footerData.designPattern}</p>` : ''}
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
