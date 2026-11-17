(() => {
  'use strict';
  const node = document.getElementById('engine-guide');
  const root = document.getElementById('guide-root');
  if (!node || !root) throw new Error('Engine guide payload is unavailable');
  const guide = JSON.parse(node.textContent || '{}');
  if (guide.schemaVersion !== 1 || guide.locale !== 'ar-SA' || !Array.isArray(guide.sections)) {
    throw new Error('Engine guide payload is invalid');
  }
  const esc = (value) => String(value).replace(/[&<>"']/g, (item) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[item]));
  const sectionMarkup = (section) => `
    <article class="guide-section" id="section-${esc(section.id)}" data-search="${esc([section.title, section.summary, ...section.facts.flatMap((fact) => [fact.label, fact.value]), ...section.examples].join(' '))}">
      <header><span>${esc(section.title)}</span><h2>${esc(section.title)}</h2><p>${esc(section.summary)}</p></header>
      <div class="facts">${section.facts.map((fact) => `<div><small>${esc(fact.label)}</small><strong>${esc(fact.value)}</strong></div>`).join('')}</div>
      ${section.tables.map((table) => `<section class="card-table"><h3>${esc(table.title)}</h3><div class="card-row">${table.cards.map((card) => `<figure><img src="${esc(card.asset)}" alt="${esc(card.rank)}" /><figcaption><b>${esc(card.rank)}</b><span>${esc(card.points)} ${esc(guide.interface.pointsLabel)}</span></figcaption></figure>`).join('')}</div></section>`).join('')}
      <section class="examples"><h3>${esc(guide.interface.examplesLabel)}</h3>${section.examples.map((example) => `<p>${esc(example)}</p>`).join('')}</section>
    </article>`;
  root.innerHTML = `
    <header class="hero">
      <nav><a href="/">${esc(guide.interface.homeLabel)}</a><a href="/foundry">${esc(guide.interface.foundryLabel)}</a></nav>
      <div class="hero-copy"><span>${esc(guide.brand)}</span><h1>${esc(guide.pageTitle)}</h1><p>${esc(guide.subtitle)}</p></div>
      <div class="profile"><small>${esc(guide.profile.label)}</small><strong>${esc(guide.profile.id)}</strong><span>${esc(guide.profile.status)} · ${esc(guide.interface.versionPrefix)} ${esc(guide.profile.version)}</span></div>
    </header>
    <main>
      <aside class="section-nav"><input id="guide-search" type="search" placeholder="${esc(guide.interface.searchPlaceholder)}" />${guide.sections.map((section) => `<a href="#section-${esc(section.id)}">${esc(section.title)}</a>`).join('')}</aside>
      <section class="content">
        <div class="boundary"><strong>${esc(guide.proofBoundary.title)}</strong><p>${esc(guide.proofBoundary.body)}</p></div>
        <div id="guide-sections">${guide.sections.map(sectionMarkup).join('')}</div>
        <p id="no-results" class="no-results" hidden>${esc(guide.interface.noResults)}</p>
        <details class="audit"><summary>${esc(guide.interface.auditLabel)}</summary><dl>${Object.entries(guide.audit).map(([key, value]) => `<div><dt>${esc(key)}</dt><dd dir="ltr">${esc(value)}</dd></div>`).join('')}<div><dt>guideHash</dt><dd dir="ltr">${esc(guide.guideHash)}</dd></div></dl></details>
      </section>
    </main>`;
  const search = document.getElementById('guide-search');
  const sections = [...document.querySelectorAll('.guide-section')];
  const noResults = document.getElementById('no-results');
  search.addEventListener('input', () => {
    const query = search.value.trim().toLocaleLowerCase('ar-SA');
    let visible = 0;
    sections.forEach((section) => {
      const match = !query || (section.dataset.search || '').toLocaleLowerCase('ar-SA').includes(query);
      section.hidden = !match;
      if (match) visible += 1;
    });
    noResults.hidden = visible !== 0;
  });
})();
