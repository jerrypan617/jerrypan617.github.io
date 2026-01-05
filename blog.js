// 博客渲染功能

// 博客数据缓存
let blogs = [];

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
        <div class="bg-gray-900 border-2 border-gray-800 p-6 text-center">
            <p class="text-neon-lime animate-pulse">Loading blog posts...</p>
        </div>
    `;
    
    // 加载博客数据
    await loadBlogs();
    
    if (blogs.length === 0) {
        blogPostsContainer.innerHTML = `
            <div class="bg-gray-900 border-2 border-gray-800 p-6 text-center">
                <p class="text-gray-400">No blog posts available yet.</p>
                <p class="text-gray-600 text-sm mt-2">Coming soon!</p>
            </div>
        `;
        return;
    }
    
    blogPostsContainer.innerHTML = `
        <div class="space-y-8">
            ${blogs.map(blog => `
                <div class="bg-gray-900 border-2 border-gray-800 p-6 hover:border-neon-lime hover:shadow-neon-lime transition-all duration-300 cursor-pointer group" onclick="openBlog('${blog.id}')">
                    <div class="flex flex-col md:flex-row md:items-center justify-between mb-3">
                        <h4 class="text-neon-cyan font-bold text-xl mb-2 md:mb-0 group-hover:text-white transition-colors">${blog.title}</h4>
                        <span class="text-xs text-gray-500 font-mono">${blog.date}</span>
                    </div>
                    <p class="text-sm text-gray-500 mb-4 font-bold">${blog.subtitle}</p>
                    
                    <div class="mb-4">
                        <p class="text-gray-300 text-sm leading-relaxed line-clamp-3">
                            ${blog.content.replace(/#+.+|```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$.+?\$/g, '').substring(0, 200)}...
                        </p>
                    </div>
                    
                    <div class="flex flex-wrap gap-2 mt-4">
                        ${blog.tags.map(tag => `
                            <span class="text-[10px] bg-gray-800 text-neon-pink px-2 py-0.5">${tag}</span>
                        `).join('')}
                    </div>
                    
                    <div class="mt-4 text-right">
                        <span class="text-xs text-neon-lime hover:underline group-hover:text-white transition-colors">
                            READ MORE →
                        </span>
                    </div>
                </div>
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
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-80 backdrop-blur-sm';
    
    // 直接渲染markdown内容，包括公式
    const renderedContent = renderMathInMarkdown(blog.content);
    
    modal.innerHTML = `
        <div class="bg-gray-900 border-2 border-neon-cyan shadow-neon-cyan max-w-4xl w-full max-h-[90vh] overflow-y-auto relative">
            <!-- Close button -->
            <button onclick="this.closest('.fixed').remove()" 
                    class="absolute top-4 right-4 text-gray-500 hover:text-neon-pink transition-colors text-xl font-bold">
                ✕
            </button>
            
            <!-- Blog header -->
            <div class="p-6 border-b border-gray-800">
                <div class="flex flex-col md:flex-row md:items-center justify-between mb-3">
                    <h2 class="text-neon-cyan font-bold text-2xl mb-2 md:mb-0">${blog.title}</h2>
                    <span class="text-xs text-gray-500 font-mono">${blog.date}</span>
                </div>
                <p class="text-sm text-gray-500 mb-4 font-bold">${blog.subtitle}</p>
                
                <div class="flex flex-wrap gap-2">
                    ${blog.tags.map(tag => `
                        <span class="text-[10px] bg-gray-800 text-neon-pink px-2 py-0.5">${tag}</span>
                    `).join('')}
                </div>
            </div>
            
            <!-- Blog content -->
            <div class="p-6 blog-content prose prose-invert max-w-none">
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
        // 添加代码高亮样式
        return `<pre class="bg-gray-800 p-6 rounded-md overflow-x-auto text-sm"><code class="language-${lang || ''}">${code}</code></pre>`;
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
    /* Blog content styles */
    .blog-content {
        color: #e0e0e0;
    }
    
    .blog-content h1 {
        color: #00f3ff;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
    }
    
    .blog-content h2 {
        color: #ccff00;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .blog-content h3 {
        color: #ff00ff;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: bold;
    }
    
    .blog-content p {
        margin-bottom: 1rem;
        line-height: 1.6;
        color: #e0e0e0;
    }
    
    .blog-content a {
        color: #00f3ff;
        text-decoration: underline;
        transition: color 0.3s;
    }
    
    .blog-content a:hover {
        color: #ccff00;
    }
    
    /* Image styles */
    .blog-content img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1.5rem auto;
        border: 2px solid #333;
        border-radius: 0.5rem;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }
    
    .blog-content ul,
    .blog-content ol {
        margin-bottom: 1rem;
        padding-left: 1.5rem;
    }
    
    .blog-content li {
        margin-bottom: 0.5rem;
        color: #e0e0e0;
    }
    
    .blog-content li::marker {
        color: #ff00ff;
    }
    
    .blog-content code {
        background-color: #1a1a1a;
        color: #ff00ff;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        font-family: 'Courier Prime', monospace;
        font-size: 0.9em;
    }
    
    .blog-content pre {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        overflow-x: auto;
        border: 1px solid #333;
    }
    
    .blog-content pre code {
        background: none;
        padding: 0;
        color: #e0e0e0;
    }
    
    .blog-content blockquote {
        border-left: 4px solid #00f3ff;
        padding-left: 1rem;
        margin-left: 0;
        margin-bottom: 1rem;
        color: #a0a0a0;
        font-style: italic;
    }
    
    .blog-content hr {
        border: none;
        border-top: 1px solid #333;
        margin: 2rem 0;
    }
    
    .blog-content table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
    }
    
    .blog-content th {
        background-color: #1a1a1a;
        color: #00f3ff;
        padding: 0.5rem;
        text-align: left;
        border: 1px solid #333;
    }
    
    .blog-content td {
        padding: 0.5rem;
        border: 1px solid #333;
        color: #e0e0e0;
    }
    
    .blog-content tr:nth-child(even) {
        background-color: #1a1a1a;
    }
    
    /* Line clamp for blog excerpts */
    .line-clamp-3 {
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
`;
document.head.appendChild(style);