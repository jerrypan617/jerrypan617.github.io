// 渲染 Header 部分
function renderHeader() {
    const header = document.getElementById('header');
    if (!header) return;

    const { personal } = siteConfig;
    
    header.innerHTML = `
        <!-- Decorative corner markers -->
        <div class="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-neon-pink"></div>
        <div class="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-neon-pink"></div>
        <div class="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-neon-pink"></div>
        <div class="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-neon-pink"></div>

        <div class="flex flex-col gap-4">
            <div class="text-neon-lime text-sm md:text-base mb-2 animate-pulse-slow">
                ${personal.bootMessage} <span class="animate-blink">_</span>
            </div>
            
            <h1 class="text-5xl md:text-7xl font-bold text-neon-cyan tracking-tighter text-glow glitch-hover cursor-pointer uppercase" onclick="window.location.href='index.html'">
                ${personal.name}
            </h1>
            
            <h2 class="text-xl md:text-2xl text-white border-l-4 border-neon-pink pl-4 py-1 mt-2">
                >>> ${personal.title}
            </h2>

            <div class="mt-6 pt-4 border-t border-gray-700 flex flex-col md:flex-row md:items-center md:justify-between gap-4 text-sm md:text-base text-gray-400 font-bold">
                <div class="flex items-center gap-2 hover:text-neon-lime transition-colors duration-300 cursor-pointer">
                    <span class="text-neon-pink">[EMAIL]</span>
                    <a href="mailto:${personal.email}" class="hover:underline">${personal.email}</a>
                </div>
                <div class="hidden md:block text-gray-600">|</div>
                <div class="flex items-center gap-2 hover:text-neon-lime transition-colors duration-300 cursor-pointer">
                    <span class="text-neon-pink">[PHONE]</span>
                    <a href="tel:${personal.phone.replace(/[()\s-]/g, '')}" class="hover:underline">${personal.phone}</a>
                </div>
            </div>

            ${(personal.social && (personal.social.x || personal.social.huggingface || personal.social.reddit)) || true ? `
                <div class="mt-4 pt-4 border-t border-gray-700">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-neon-pink text-xs font-bold">[SOCIAL_NETWORKS]</span>
                    </div>
                    <div class="flex flex-wrap gap-3">
                        <a href="blog.html" 
                           class="inline-flex items-center gap-1.5 text-xs text-gray-400 hover:text-neon-lime transition-colors border border-gray-700 hover:border-neon-lime px-3 py-1.5 hover:bg-neon-lime hover:bg-opacity-10 group">
                            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                            </svg>
                            <span class="group-hover:underline">Blog</span>
                        </a>
                        ${personal.social.x ? `
                            <a href="${personal.social.x}" target="_blank" rel="noopener noreferrer" 
                               class="inline-flex items-center gap-1.5 text-xs text-gray-400 hover:text-neon-cyan transition-colors border border-gray-700 hover:border-neon-cyan px-3 py-1.5 hover:bg-neon-cyan hover:bg-opacity-10 group">
                                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                                </svg>
                                <span class="group-hover:underline">X</span>
                            </a>
                        ` : ''}
                        ${personal.social.huggingface ? `
                            <a href="${personal.social.huggingface}" target="_blank" rel="noopener noreferrer" 
                               class="inline-flex items-center gap-1.5 text-xs text-gray-400 hover:text-neon-lime transition-colors border border-gray-700 hover:border-neon-lime px-3 py-1.5 hover:bg-neon-lime hover:bg-opacity-10 group">
                                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 2.4c5.302 0 9.6 4.298 9.6 9.6 0 5.302-4.298 9.6-9.6 9.6-5.302 0-9.6-4.298-9.6-9.6 0-5.302 4.298-9.6 9.6-9.6zm-2.4 4.8c-1.325 0-2.4 1.075-2.4 2.4s1.075 2.4 2.4 2.4 2.4-1.075 2.4-2.4-1.075-2.4-2.4-2.4zm4.8 0c-1.325 0-2.4 1.075-2.4 2.4s1.075 2.4 2.4 2.4 2.4-1.075 2.4-2.4-1.075-2.4-2.4-2.4zm-2.4 7.2c-2.65 0-4.8 1.35-4.8 3v1.2h9.6v-1.2c0-1.65-2.15-3-4.8-3z"/>
                                </svg>
                                <span class="group-hover:underline">Hugging Face</span>
                            </a>
                        ` : ''}
                        ${personal.social.reddit ? `
                            <a href="${personal.social.reddit}" target="_blank" rel="noopener noreferrer" 
                               class="inline-flex items-center gap-1.5 text-xs text-gray-400 hover:text-neon-pink transition-colors border border-gray-700 hover:border-neon-pink px-3 py-1.5 hover:bg-neon-pink hover:bg-opacity-10 group">
                                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                                </svg>
                                <span class="group-hover:underline">Reddit</span>
                            </a>
                        ` : ''}
                    </div>
                </div>
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
        <div class="border border-gray-700 bg-gray-900/50 p-6 md:p-8 relative hover:border-neon-lime transition-colors duration-300 group">
            <!-- Top bar graphic -->
            <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-neon-lime via-transparent to-neon-lime opacity-50"></div>
            
            <div class="grid md:grid-cols-[1fr_auto] gap-6 items-start">
                <div class="space-y-4">
                    ${profileData.education && profileData.education.length > 0 ? `
                        <div class="space-y-3">
                            ${profileData.education.map((edu, index) => `
                                <div class="bg-gray-800/50 p-3 border-l-2 ${index === 0 ? 'border-neon-cyan' : 'border-neon-lime'}">
                                    <p class="text-gray-300 leading-relaxed">
                                        <span class="text-neon-pink font-bold">[${edu.degree}]</span> 
                                        <span class="text-neon-cyan font-bold">${edu.major}</span> 
                                        at <span class="text-white">${edu.university}</span>
                                        ${edu.period ? `<span class="text-gray-500 text-sm ml-2">(${edu.period})</span>` : ''}
                                    </p>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                    <p class="text-gray-400 pt-2">
                        <span class="text-neon-pink">>> FOCUS:</span> ${profileData.focus}.
                    </p>
                </div>
                
                <!-- Decorative status block -->
                <div class="hidden md:flex flex-col gap-1 text-[10px] text-neon-lime opacity-70 font-bold text-right border-r border-neon-lime pr-2">
                    <span>STATUS: ONLINE</span>
                    <span>MEM: 64GB OK</span>
                    <span>CPU: 98% LOAD</span>
                    <span>NET: SECURE</span>
                </div>
            </div>
        </div>
    `;
}

// 渲染 Skills 部分
function renderSkills() {
    const skillsContainer = document.getElementById('skills');
    if (!skillsContainer) return;

    const { skills } = siteConfig;
    
    skillsContainer.innerHTML = `
        <div class="grid gap-6">
            <!-- Programming -->
            <div class="bg-gray-900 border border-gray-700 p-4 relative overflow-hidden group hover:shadow-neon-cyan hover:border-neon-cyan transition-all duration-300">
                <div class="absolute top-0 right-0 bg-gray-800 text-xs text-gray-400 px-2 py-1 border-b border-l border-gray-700 font-bold">
                    LIB.01
                </div>
                <h4 class="text-neon-pink font-bold mb-3 uppercase tracking-wider text-sm">Programming</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.programming.map(skill => `
                        <span class="px-3 py-1 bg-gray-800 border border-neon-cyan text-neon-cyan text-sm font-bold hover:bg-neon-cyan hover:text-black transition-colors cursor-default">
                            ${skill}
                        </span>
                    `).join('')}
                </div>
            </div>

            <!-- Frameworks -->
            <div class="bg-gray-900 border border-gray-700 p-4 relative overflow-hidden group hover:shadow-neon-cyan hover:border-neon-cyan transition-all duration-300">
                <div class="absolute top-0 right-0 bg-gray-800 text-xs text-gray-400 px-2 py-1 border-b border-l border-gray-700 font-bold">
                    LIB.02
                </div>
                <h4 class="text-neon-pink font-bold mb-3 uppercase tracking-wider text-sm">Frameworks / Libraries</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.frameworks.map(skill => `
                        <span class="px-3 py-1 bg-gray-800 border border-gray-600 text-gray-300 text-sm hover:border-neon-pink hover:text-neon-pink transition-colors cursor-default">${skill}</span>
                    `).join('')}
                </div>
            </div>

            <!-- Domains -->
            <div class="bg-gray-900 border border-gray-700 p-4 relative overflow-hidden group hover:shadow-neon-cyan hover:border-neon-cyan transition-all duration-300">
                <div class="absolute top-0 right-0 bg-gray-800 text-xs text-gray-400 px-2 py-1 border-b border-l border-gray-700 font-bold">
                    LIB.03
                </div>
                <h4 class="text-neon-pink font-bold mb-3 uppercase tracking-wider text-sm">Research Domains</h4>
                <div class="flex flex-wrap gap-2">
                    ${skills.domains.map(domain => `
                        <span class="px-3 py-1 bg-gray-800 border-l-4 border-neon-lime text-gray-200 text-sm">${domain}</span>
                    `).join('')}
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
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            ${projects.map((project, index) => {
                const typeColor = project.type === 'RESEARCH' ? 'neon-lime' : 'neon-cyan';
                return `
                    <div class="bg-gray-900 border-2 border-gray-800 p-5 flex flex-col h-full hover:border-neon-lime hover:shadow-neon-lime transition-all duration-300 group relative">
                        <div class="absolute top-2 right-2 text-[10px] text-${typeColor} opacity-50 border border-${typeColor} px-1">${project.type}</div>
                        <h4 class="text-neon-cyan font-bold text-lg mb-2 mt-4 group-hover:text-white transition-colors">${project.name}</h4>
                        <p class="text-xs text-gray-500 mb-3 font-bold">${project.subtitle}</p>
                        
                        <div class="mb-4 flex-grow">
                            <p class="text-gray-300 text-sm leading-relaxed border-l-2 border-gray-700 pl-3 group-hover:border-neon-lime transition-colors">
                                ${project.description}
                            </p>
                        </div>
                        
                        <div class="mt-auto space-y-2">
                            <div class="pt-3 border-t border-gray-800 flex flex-wrap gap-1">
                                ${project.technologies.map(tech => `
                                    <span class="text-[10px] bg-gray-800 text-neon-pink px-2 py-0.5">${tech}</span>
                                `).join('')}
                            </div>
                            ${(project.github || project.paper) ? `
                                <div class="flex flex-wrap gap-2">
                                    ${project.github ? `
                                        <a href="${project.github}" target="_blank" rel="noopener noreferrer" 
                                           class="inline-flex items-center gap-1 text-xs text-neon-cyan hover:text-neon-lime transition-colors border border-neon-cyan px-2 py-1 hover:bg-neon-cyan hover:bg-opacity-10 group">
                                            <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                                            </svg>
                                            <span class="group-hover:underline">GitHub</span>
                                        </a>
                                    ` : ''}
                                    ${project.paper ? `
                                        <a href="${project.paper}" target="_blank" rel="noopener noreferrer" 
                                           class="inline-flex items-center gap-1 text-xs text-neon-pink hover:text-neon-lime transition-colors border border-neon-pink px-2 py-1 hover:bg-neon-pink hover:bg-opacity-10 group">
                                            <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                                <path d="M14,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V8L14,3M19,19H5V5H13V9H19M17,17H7V15H17M17,13H7V11H17M17,9H7V7H17V9Z"/>
                                            </svg>
                                            <span class="group-hover:underline">Paper</span>
                                        </a>
                                    ` : ''}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// 渲染 Achievements 部分
function renderAchievements() {
    const achievementsContainer = document.getElementById('achievements');
    if (!achievementsContainer) return;

    const { achievements } = siteConfig;
    
    achievementsContainer.innerHTML = `
        <div class="space-y-3">
            ${achievements.map(achievement => `
                <div class="flex flex-col md:flex-row md:items-center justify-between bg-gray-900/80 p-4 border-l-4 border-${achievement.color} hover:bg-gray-800 transition-colors cursor-default group border-y border-r border-gray-800 md:border-y-0 md:border-r-0">
                    <div class="flex items-start gap-3">
                        <span class="text-${achievement.color} mt-1 md:mt-0">></span>
                        <div>
                            <span class="text-gray-200 font-bold group-hover:text-neon-cyan transition-colors">${achievement.title}</span>
                            ${achievement.subtitle ? `<span class="block text-xs text-${achievement.color}">${achievement.subtitle}</span>` : ''}
                        </div>
                    </div>
                    <span class="text-xs text-gray-500 font-mono mt-1 md:mt-0 md:text-right">${achievement.date}</span>
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
        <p>SYSTEM_ID: ${footerData.systemId} // END_OF_LINE</p>
        <p class="mt-2 opacity-50">DESIGN_PATTERN: ${footerData.designPattern}</p>
    `;
}

// 初始化：页面加载完成后渲染所有内容
document.addEventListener('DOMContentLoaded', function() {
    // 渲染所有可能存在的组件
    renderHeader();
    
    // 只在存在对应元素时渲染
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

