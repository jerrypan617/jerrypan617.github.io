// 博客渲染功能

let blogs = [];

/**
 * 与 blog.html / index.html 同级的站点根 URL（末尾带 /）。
 * 支持 https://user.github.io/blog.html 与 https://user.github.io/repo/blog.html
 */
function getSiteBaseUrl() {
    if (typeof window === 'undefined') return '/';
    const u = new URL(window.location.href);
    const segs = u.pathname.split('/').filter(Boolean);
    segs.pop();
    const path = segs.length ? `/${segs.join('/')}/` : '/';
    return `${u.origin}${path}`;
}

function blogFileUrl(filename) {
    const base = getSiteBaseUrl();
    const path = `blogs/${filename}`.replace(/\/{2,}/g, '/');
    return new URL(path, base).href;
}

function normalizeTags(tags) {
    if (!tags) return [];
    return Array.isArray(tags) ? tags : [tags];
}

// 解析YAML Front Matter
function parseFrontMatter(content) {
    const frontMatterRegex = /^---\s*([\s\S]*?)\s*---/;
    const match = content.match(frontMatterRegex);
    
    if (!match) {
        return { metadata: {}, content: content };
    }
    
    const frontMatter = match[1];
    const restContent = content.replace(frontMatterRegex, '').trim();
    
    // 简单的YAML解析（仅支持键值对和数组）
    const metadata = {};
    const lines = frontMatter.split('\n');
    
    for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine || trimmedLine.startsWith('#')) continue;
        
        const colonIndex = trimmedLine.indexOf(':');
        if (colonIndex === -1) continue;
        
        const key = trimmedLine.substring(0, colonIndex).trim();
        let value = trimmedLine.substring(colonIndex + 1).trim();
        
        // 处理数组
        if (value.startsWith('[') && value.endsWith(']')) {
            value = value.substring(1, value.length - 1).split(',').map(item => item.trim().replace(/['"]/g, ''));
        }
        // 处理字符串
        else if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.substring(1, value.length - 1);
        }
        
        metadata[key] = value;
    }
    
    return { metadata, content: restContent };
}

// 获取博客文件列表
async function getBlogFiles() {
    try {
        const response = await fetch(new URL('blogs/', getSiteBaseUrl()).href);
        if (!response.ok) return [];
        const html = await response.text();
        
        // 从HTML中提取.md文件链接
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const links = doc.querySelectorAll('a[href$=".md"]');
        
        return Array.from(links).map(link => link.href.split('/').pop());
    } catch (error) {
        console.error('Failed to get blog files:', error);
        return [];
    }
}

// 读取并解析单个博客文件（仓库根目录需有 .nojekyll，否则 GitHub Pages 上 Jekyll 可能不发布原始 .md）
async function readBlogFile(filename) {
    const url = blogFileUrl(filename);
    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.error(
                `[blog] ${response.status} ${filename}\n  请求: ${url}\n  若线上 404：确认已 push，且仓库根目录存在 .nojekyll（禁用 Jekyll）后重新部署 Pages。`
            );
            return null;
        }
        const content = await response.text();
        if (!content.trim() || content.trim().startsWith('<!')) {
            console.error(`Blog ${filename}: 期望 Markdown，实际为 HTML 或空（可能被 Jekyll 改成网页）。请保留仓库根目录的 .nojekyll 文件。`);
            return null;
        }

        const { metadata, content: markdownContent } = parseFrontMatter(content);

        return {
            id: filename.replace('.md', ''),
            filename: filename,
            title: metadata.title || 'Untitled',
            subtitle: metadata.subtitle || '',
            date: metadata.date || 'Unknown Date',
            tags: metadata.tags || [],
            content: markdownContent
        };
    } catch (error) {
        console.error(`Failed to read blog file ${filename} (${url}):`, error);
        return null;
    }
}

const DEFAULT_BLOG_FILES = ['attention-sink-analysis.md'];

// 加载所有博客（优先 siteConfig.blogFiles；GitHub Pages 无目录索引时 getBlogFiles 常为空，故用默认列表兜底）
async function loadBlogs() {
    let blogFiles = [];
    if (typeof siteConfig !== 'undefined' && Array.isArray(siteConfig.blogFiles) && siteConfig.blogFiles.length > 0) {
        blogFiles = [...siteConfig.blogFiles];
    } else {
        blogFiles = await getBlogFiles();
    }
    if (blogFiles.length === 0) {
        blogFiles = [...DEFAULT_BLOG_FILES];
    }

    const blogPromises = blogFiles.map((file) => readBlogFile(file));
    const loadedBlogs = await Promise.all(blogPromises);

    blogs = loadedBlogs
        .filter((blog) => blog !== null)
        .sort((a, b) => new Date(b.date) - new Date(a.date));

    return blogs;
}

// 渲染博客列表
async function renderBlogList() {
    const blogPostsContainer = document.getElementById('blog-posts');
    if (!blogPostsContainer) return;
    
    // 显示加载状态
    blogPostsContainer.innerHTML = `
        <div class="py-6 text-center">
            <p class="text-sm text-zinc-600">Loading…</p>
        </div>
    `;
    
    // 加载博客数据
    await loadBlogs();
    
    if (blogs.length === 0) {
        blogPostsContainer.innerHTML = `
            <div class="py-6 text-center">
                <p class="text-zinc-600">No posts yet.</p>
            </div>
        `;
        return;
    }
    
    blogPostsContainer.innerHTML = `
        <div class="space-y-0 divide-y divide-zinc-200 reveal">
            ${blogs.map(blog => `
                <article class="cursor-pointer group py-3.5 sm:py-4 first:pt-0 active:bg-zinc-100 sm:active:bg-transparent -mx-1 px-1 sm:mx-0 sm:px-0 rounded-lg sm:rounded-none" onclick="openBlog('${blog.id}')">
                    <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1.5 mb-1.5">
                        <h4 class="text-[15px] sm:text-base font-semibold text-zinc-900 sm:group-hover:text-emerald-700 transition-colors tracking-tight break-words pr-1">${blog.title}</h4>
                        <span class="text-[11px] text-zinc-600 tabular-nums shrink-0">${blog.date}</span>
                    </div>
                    ${blog.subtitle ? `<p class="text-xs text-zinc-600 mb-2">${blog.subtitle}</p>` : ''}
                    ${normalizeTags(blog.tags).length ? `
                        <div class="flex flex-wrap gap-2">
                            ${normalizeTags(blog.tags).map(tag => `
                                <span class="text-[11px] font-medium text-zinc-600 px-2 py-0.5 rounded-md bg-zinc-100 border border-zinc-200">${tag}</span>
                            `).join('')}
                        </div>
                    ` : ''}
                    <p class="text-[11px] font-medium text-emerald-700 mt-2 opacity-90 group-hover:opacity-100 transition-opacity">Read article →</p>
                </article>
            `).join('')}
        </div>
    `;
}

// 关闭弹窗（带动画）
function closeModal(modal) {
    if (!modal || modal.classList.contains('closing')) return;
    modal.classList.add('closing');
    // 清理 ESC 键盘监听
    if (modal._escHandler) {
        document.removeEventListener('keydown', modal._escHandler);
    }
    modal.addEventListener('animationend', () => modal.remove(), { once: true });
}

// 打开博客详情
async function openBlog(blogId) {
    const blog = blogs.find(b => b.id === blogId);
    if (!blog) return;

    // 创建博客详情模态框
    const modal = document.createElement('div');
    modal.className = 'modal-overlay fixed inset-0 z-50 flex items-stretch sm:items-center justify-center p-0 sm:p-4 bg-zinc-900/40 sm:bg-zinc-900/35 backdrop-blur-sm';

    // 直接渲染markdown内容，包括公式
    const renderedContent = renderMathInMarkdown(blog.content);

    modal.innerHTML = `
        <div class="modal-content bg-white border-0 sm:border border-zinc-200 max-w-none sm:max-w-3xl w-full h-full sm:h-auto max-h-none sm:max-h-[90vh] min-h-0 overflow-y-auto overscroll-contain relative rounded-none sm:rounded-xl shadow-none sm:shadow-xl sm:shadow-zinc-300/40" onclick="event.stopPropagation()">
            <div class="sticky top-0 z-20 flex gap-3 items-start p-4 sm:p-5 md:p-6 border-b border-zinc-200 bg-white backdrop-blur-sm">
                <div class="min-w-0 flex-1 space-y-2">
                    <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1.5 sm:gap-2">
                        <h2 class="text-base sm:text-lg font-semibold text-zinc-900 tracking-tight break-words">${blog.title}</h2>
                        <span class="text-[11px] sm:text-xs text-zinc-600 tabular-nums shrink-0">${blog.date}</span>
                    </div>
                    ${blog.subtitle ? `<p class="text-xs text-zinc-600">${blog.subtitle}</p>` : ''}
                    ${normalizeTags(blog.tags).length ? `
                        <div class="flex flex-wrap gap-2">
                            ${normalizeTags(blog.tags).map(tag => `
                                <span class="text-[11px] font-medium text-zinc-600 px-2 py-0.5 rounded-md bg-zinc-100 border border-zinc-200">${tag}</span>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
                <button type="button" onclick="closeModal(this.closest('.modal-overlay'))"
                        class="shrink-0 -mr-1 text-zinc-500 hover:text-zinc-900 transition-colors text-sm w-11 h-11 sm:w-8 sm:h-8 flex items-center justify-center rounded-lg hover:bg-zinc-100" aria-label="Close">
                    ✕
                </button>
            </div>
            <div class="p-4 sm:p-5 md:p-6 pb-[max(2rem,env(safe-area-inset-bottom))] sm:pb-6 blog-content max-w-none">
                ${renderedContent}
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // 点击外部关闭（带动画）
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal(modal);
    });

    // ESC 关闭（handler 挂载在 modal 上便于 closeModal 清理）
    modal._escHandler = (e) => { if (e.key === 'Escape') closeModal(modal); };
    document.addEventListener('keydown', modal._escHandler);
}

// 配置marked选项，确保HTML被正确处理
marked.use({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        return `<pre class="bg-zinc-900 p-4 border border-zinc-700 overflow-x-auto text-sm rounded-lg"><code class="language-${lang || ''} text-zinc-200">${code}</code></pre>`;
    }
});

// 自定义函数，直接渲染公式
function renderMathInMarkdown(content) {
    // 1. 先用占位符保护 LaTeX 公式，避免 marked 把 _ 解析成 <em>
    const placeholders = [];

    // 保护块级公式 $$...$$
    let protectedContent = content.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
        const ph = `%%MATH_BLOCK_${placeholders.length}%%`;
        placeholders.push({ type: 'block', content: math.trim(), ph });
        return ph;
    });
    // 保护行内公式 $...$
    protectedContent = protectedContent.replace(/\$([^$\n]+?)\$/g, (_, math) => {
        const ph = `%%MATH_INLINE_${placeholders.length}%%`;
        placeholders.push({ type: 'inline', content: math.trim(), ph });
        return ph;
    });

    // 2. marked 解析 markdown（此时 LaTeX 已是安全占位符）
    let html = marked.parse(protectedContent, { breaks: false, gfm: true });

    // 修复图片路径：blog 内容渲染在页面根路径下，需补上 blogs/ 前缀
    html = html.replace(/(<img[^>]+src=")(?!https?:\/\/|\/)([^"]+)"/gi, (_, prefix, path) => {
        return `${prefix}blogs/${path}"`;
    });

    // 3. 将占位符替换为 KaTeX 渲染结果
    for (const { type, content, ph } of placeholders) {
        try {
            const rendered = katex.renderToString(content, {
                displayMode: type === 'block',
                throwOnError: false
            });
            html = html.replace(ph, rendered);
        } catch (err) {
            console.error(`Error rendering ${type} math:`, err);
        }
    }

    return html;
}

// 初始化：仅在独立 blog.html 页面时自动执行
document.addEventListener('DOMContentLoaded', function() {
    if (!/blog\.html$/i.test(window.location.pathname)) return;
    renderBlogList().then(() => {
        if (typeof observeReveal === 'function') observeReveal();
    });
    renderHeader();
    renderFooter();
});

// Export closeModal globally for inline onclick
window.closeModal = closeModal;

// 添加CSS样式用于博客内容
const style = document.createElement('style');
style.textContent = `
    .blog-content {
        color: #52525b;
        font-size: 0.875rem;
        line-height: 1.65;
    }

    .blog-content h1 {
        color: #18181b;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
        margin-top: 0;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid #e4e4e7;
        letter-spacing: -0.02em;
    }

    .blog-content h2 {
        color: #18181b;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.35rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .blog-content h3 {
        color: #3f3f46;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1.1rem;
        margin-bottom: 0.35rem;
    }

    .blog-content p {
        margin-bottom: 0.75rem;
    }

    .blog-content a {
        color: #0d9f7a;
        text-decoration: none;
        border-bottom: 1px solid rgba(13, 159, 122, 0.35);
        transition: color 0.15s, border-color 0.15s;
    }

    .blog-content a:hover {
        color: #0a7a5e;
        border-bottom-color: rgba(10, 122, 94, 0.55);
    }

    .blog-content .katex,
    .blog-content .katex * {
        color: #18181b !important;
    }

    .blog-content img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1.5rem 0;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
    }

    .blog-content ul,
    .blog-content ol {
        margin-bottom: 1rem;
        padding-left: 1.25rem;
    }

    .blog-content li {
        margin-bottom: 0.35rem;
    }

    .blog-content li::marker {
        color: #71717a;
    }

    .blog-content code {
        background-color: #f4f4f5;
        color: #18181b;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.875em;
        border: 1px solid #e4e4e7;
    }

    .blog-content pre {
        background-color: #18181b;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        overflow-x: auto;
        border: 1px solid #3f3f46;
    }

    .blog-content pre code {
        background: none;
        padding: 0;
        border: none;
        color: #e4e4e7;
    }

    .blog-content blockquote {
        border-left: 3px solid rgba(13, 159, 122, 0.5);
        padding-left: 1rem;
        margin: 1rem 0;
        color: #71717a;
    }

    .blog-content hr {
        border: none;
        border-top: 1px solid #e4e4e7;
        margin: 1.35rem 0;
    }

    .blog-content table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
        font-size: 0.875rem;
    }

    .blog-content th {
        background-color: #f4f4f5;
        color: #18181b;
        font-weight: 600;
        padding: 0.5rem 0.75rem;
        text-align: left;
        border: 1px solid #e4e4e7;
    }

    .blog-content td {
        padding: 0.5rem 0.75rem;
        border: 1px solid #e4e4e7;
        color: #52525b;
    }

    .blog-content tr:nth-child(even) {
        background-color: #fafafa;
    }

    .line-clamp-3 {
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
`;
document.head.appendChild(style);