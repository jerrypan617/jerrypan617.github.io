/* GA4：仅在 config.js 中配置了 ga4MeasurementId 时加载 */
(function () {
    if (typeof siteConfig === 'undefined') return;
    var id = siteConfig.analytics && siteConfig.analytics.ga4MeasurementId;
    if (!id || typeof id !== 'string') return;
    id = id.trim();
    if (!id.startsWith('G-')) return;

    window.dataLayer = window.dataLayer || [];
    function gtag() {
        window.dataLayer.push(arguments);
    }
    window.gtag = gtag;
    gtag('js', new Date());
    gtag('config', id);

    var s = document.createElement('script');
    s.async = true;
    s.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(id);
    document.head.appendChild(s);
})();
