// 博客渲染功能

let blogs = [];

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
        const response = await fetch('/blogs/');
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

// 读取并解析单个博客文件
async function readBlogFile(filename) {
    try {
        const response = await fetch(`/blogs/${filename}`);
        const content = await response.text();
        
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
        console.error(`Failed to read blog file ${filename}:`, error);
        return null;
    }
}

// 加载所有博客
async function loadBlogs() {
    const blogFiles = await getBlogFiles();
    const blogPromises = blogFiles.map(file => readBlogFile(file));
    const loadedBlogs = await Promise.all(blogPromises);
    
    // 过滤掉加载失败的博客，并按日期倒序排序
    blogs = loadedBlogs.filter(blog => blog !== null)
                      .sort((a, b) => new Date(b.date) - new Date(a.date));
    
    return blogs;
}

// 渲染博客列表
async function renderBlogList() {
    const blogPostsContainer = document.getElementById('blog-posts');
    if (!blogPostsContainer) return;
    
    // 显示加载状态
    blogPostsContainer.innerHTML = `
        <div class="py-12 text-center">
            <p class="text-sm text-neutral-500">Loading…</p>
        </div>
    `;
    
    // 加载博客数据
    await loadBlogs();
    
    if (blogs.length === 0) {
        blogPostsContainer.innerHTML = `
            <div class="py-12 text-center">
                <p class="text-neutral-600">No posts yet.</p>
            </div>
        `;
        return;
    }
    
    blogPostsContainer.innerHTML = `
        <div class="space-y-10">
            ${blogs.map(blog => `
                <article class="cursor-pointer group border-b border-neutral-200 pb-10 last:border-0 last:pb-0" onclick="openBlog('${blog.id}')">
                    <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2 mb-2">
                        <h4 class="text-lg font-medium text-neutral-900 group-hover:text-neutral-600 transition-colors">${blog.title}</h4>
                        <span class="text-xs text-neutral-400 shrink-0">${blog.date}</span>
                    </div>
                    ${blog.subtitle ? `<p class="text-sm text-neutral-500 mb-3">${blog.subtitle}</p>` : ''}
                    <p class="text-sm text-neutral-600 leading-relaxed line-clamp-3 mb-4">
                        ${blog.content.replace(/#+.+|```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$.+?\$/g, '').substring(0, 200)}…
                    </p>
                    ${normalizeTags(blog.tags).length ? `
                        <div class="flex flex-wrap gap-2">
                            ${normalizeTags(blog.tags).map(tag => `
                                <span class="text-xs text-neutral-400">${tag}</span>
                            `).join('')}
                        </div>
                    ` : ''}
                    <p class="text-xs text-neutral-400 mt-4 group-hover:text-neutral-600 transition-colors">Read</p>
                </article>
            `).join('')}
        </div>
    `;
}

// 打开博客详情
async function openBlog(blogId) {
    const blog = blogs.find(b => b.id === blogId);
    if (!blog) return;
    
    // 创建博客详情模态框
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4 bg-neutral-900/40 backdrop-blur-[2px]';
    
    // 直接渲染markdown内容，包括公式
    const renderedContent = renderMathInMarkdown(blog.content);
    
    modal.innerHTML = `
        <div class="bg-white border border-neutral-200 max-w-3xl w-full max-h-[90vh] overflow-y-auto relative shadow-lg shadow-neutral-900/5">
            <button type="button" onclick="this.closest('.fixed').remove()"
                    class="absolute top-4 right-4 text-neutral-400 hover:text-neutral-900 transition-colors text-sm w-8 h-8 flex items-center justify-center" aria-label="Close">
                ✕
            </button>
            <div class="p-8 md:p-10 border-b border-neutral-100 pr-14">
                <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2 mb-2">
                    <h2 class="text-xl font-semibold text-neutral-900">${blog.title}</h2>
                    <span class="text-xs text-neutral-400 shrink-0">${blog.date}</span>
                </div>
                ${blog.subtitle ? `<p class="text-sm text-neutral-500 mb-4">${blog.subtitle}</p>` : ''}
                ${normalizeTags(blog.tags).length ? `
                    <div class="flex flex-wrap gap-2">
                        ${normalizeTags(blog.tags).map(tag => `
                            <span class="text-xs text-neutral-400">${tag}</span>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
            <div class="p-8 md:p-10 blog-content prose max-w-none">
                ${renderedContent}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 添加模态框关闭事件（点击外部关闭）
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// 配置marked选项，确保HTML被正确处理
marked.use({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        return `<pre class="bg-neutral-50 p-4 border border-neutral-200 overflow-x-auto text-sm rounded-sm"><code class="language-${lang || ''}">${code}</code></pre>`;
    }
});

// 自定义函数，直接渲染公式
function renderMathInMarkdown(content) {
    // 1. 首先使用marked解析markdown，但禁用breaks选项
    // 这样不会将所有换行都转换为<br>标签
    const html = marked.parse(content, {
        breaks: false,  // 关键：禁用自动转换换行为<br>
        gfm: true
    });
    
    // 2. 处理块级公式
    let finalHtml = html;
    let blockMatch;
    const blockRegex = /\$\$(.*?)\$\$/gs;
    
    while ((blockMatch = blockRegex.exec(finalHtml)) !== null) {
        const original = blockMatch[0];
        const mathContent = blockMatch[1].trim();
        
        try {
            // 使用katex渲染块级公式
            const rendered = katex.renderToString(mathContent, {
                displayMode: true,
                throwOnError: false
            });
            finalHtml = finalHtml.replace(original, rendered);
        } catch (error) {
            console.error('Error rendering block math:', error);
        }
    }
    
    // 3. 处理行内公式
    let inlineMatch;
    const inlineRegex = /\$(.*?)\$/g;
    
    while ((inlineMatch = inlineRegex.exec(finalHtml)) !== null) {
        const original = inlineMatch[0];
        const mathContent = inlineMatch[1].trim();
        
        try {
            // 使用katex渲染行内公式
            const rendered = katex.renderToString(mathContent, {
                displayMode: false,
                throwOnError: false
            });
            finalHtml = finalHtml.replace(original, rendered);
        } catch (error) {
            console.error('Error rendering inline math:', error);
        }
    }
    
    return finalHtml;
}

// 初始化：页面加载完成后渲染博客列表
document.addEventListener('DOMContentLoaded', function() {
    renderBlogList();
    // 确保header和footer也被渲染
    renderHeader();
    renderFooter();
});

// 添加CSS样式用于博客内容
const style = document.createElement('style');
style.textContent = `
    .blog-content {
        color: #404040;
        font-size: 0.9375rem;
        line-height: 1.7;
    }

    .blog-content h1 {
        color: #171717;
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        margin-top: 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e5e5;
        letter-spacing: -0.02em;
    }

    .blog-content h2 {
        color: #171717;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }

    .blog-content h3 {
        color: #262626;
        font-size: 1.0625rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .blog-content p {
        margin-bottom: 1rem;
    }

    .blog-content a {
        color: #171717;
        text-decoration: underline;
        text-underline-offset: 3px;
        text-decoration-color: #d4d4d4;
    }

    .blog-content a:hover {
        text-decoration-color: #a3a3a3;
    }

    .blog-content img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1.5rem 0;
        border: 1px solid #e5e5e5;
        border-radius: 2px;
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
        color: #a3a3a3;
    }

    .blog-content code {
        background-color: #f5f5f5;
        color: #171717;
        padding: 0.15rem 0.35rem;
        border-radius: 2px;
        font-size: 0.875em;
    }

    .blog-content pre {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 2px;
        margin-bottom: 1rem;
        overflow-x: auto;
        border: 1px solid #e5e5e5;
    }

    .blog-content pre code {
        background: none;
        padding: 0;
        color: #404040;
    }

    .blog-content blockquote {
        border-left: 2px solid #e5e5e5;
        padding-left: 1rem;
        margin: 1rem 0;
        color: #737373;
    }

    .blog-content hr {
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 2rem 0;
    }

    .blog-content table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
        font-size: 0.875rem;
    }

    .blog-content th {
        background-color: #fafafa;
        color: #171717;
        font-weight: 600;
        padding: 0.5rem 0.75rem;
        text-align: left;
        border: 1px solid #e5e5e5;
    }

    .blog-content td {
        padding: 0.5rem 0.75rem;
        border: 1px solid #e5e5e5;
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