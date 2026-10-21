/** Render the offline HTML to PDF and record layout evidence for the release. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES ?? process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES
  ?? path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const require=createRequire(path.join(runtime,'__runtime__.cjs'));
const { chromium }=require('playwright');
const build=path.join(root,'.local/release_build/html');
await fs.mkdir(build,{recursive:true});
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
  await page.goto(pathToFileURL(path.join(root,'docs/submission/research_showcase.html')).href);
  await page.evaluate(()=>document.fonts.ready);
  const expectedSlides=10;
  const status=page.locator('.deck-current');
  if((await status.textContent())!=='Section 1 of 10') throw new Error('Presentation did not open on section 1 of 10.');
  await page.getByRole('button',{name:'Next section'}).click();
  if((await status.textContent())!=='Section 2 of 10') throw new Error('Next button did not advance one section.');
  await page.keyboard.press('ArrowRight');
  if((await status.textContent())!=='Section 3 of 10') throw new Error('Right arrow did not advance one section.');
  await page.keyboard.press('ArrowLeft');
  if((await status.textContent())!=='Section 2 of 10') throw new Error('Left arrow did not return one section.');
  await page.getByRole('button',{name:'Previous section'}).click();
  if((await status.textContent())!=='Section 1 of 10') throw new Error('Previous button did not return to section 1.');
  for(let i=0;i<expectedSlides;i++){
    await page.screenshot({path:path.join(build,`deck-${String(i+1).padStart(2,'0')}.png`)});
    if(i<expectedSlides-1) await page.getByRole('button',{name:'Next section'}).click();
  }
  await page.getByRole('button',{name:'Previous section'}).click();
  await page.locator('.reading-toggle').click();
  if(!await page.locator('#original').isVisible()) throw new Error('Reading view did not reveal the full document.');
  await page.locator('.reading-toggle').click();
  if((await status.textContent())!=='Section 9 of 10') throw new Error('Presentation mode did not restore the active section.');
  await page.getByRole('button',{name:'Previous section'}).click();
  await page.screenshot({path:path.join(build,'desktop.png'),fullPage:true});
  await page.screenshot({path:path.join(build,'desktop-fold.png'),fullPage:false});
  const observations=[];
  for(const width of [320,375,414,768,1280,1920]){
    await page.setViewportSize({width,height:960});
    await page.evaluate(()=>document.fonts.ready);
    observations.push(await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
      missingImages:[...document.images].filter(x=>!x.complete||!x.naturalWidth).length,
      wrappedNavigation:[...document.querySelectorAll('nav a,.downloads a,.colophon a')]
        .filter(e=>{const r=document.createRange();r.selectNodeContents(e);return r.getClientRects().length>1;}).map(e=>e.textContent)})));
    if(width!==1280) await page.screenshot({path:path.join(build,`width-${width}.png`),fullPage:true});
    if(width===375) await page.screenshot({path:path.join(build,'mobile-fold.png'),fullPage:false});
  }
  await page.setViewportSize({width:1280,height:960});
  await page.pdf({path:path.join(root,'docs/submission/research_showcase.pdf'),format:'A4',printBackground:true,
    preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',
    footerTemplate:'<div style="font-size:8px;width:100%;text-align:center">Conditional Hybrid Retrieval / completed experimental scope / <span class="pageNumber"></span> of <span class="totalPages"></span></div>'});
  await fs.writeFile(path.join(build,'layout_observations.json'),JSON.stringify(observations,null,2));
  console.log(JSON.stringify(observations,null,2));
  if(observations.some(x=>x.scrollWidth>x.width || x.missingImages || x.wrappedNavigation.length))
    throw new Error('Fix HTML layout before delivery.');
  console.log('Showcase PDF exported.');
} finally { await browser.close(); }
