const state = {
  query: "",
  degree: "",
  cities: new Set(),
  skills: new Set(),
  minYears: 0,
  results: [],
  debounceTimer: null,
  requestSeq: 0,
};

const els = {
  searchInput: document.querySelector("#searchInput"),
  searchButton: document.querySelector("#searchButton"),
  yearsRange: document.querySelector("#yearsRange"),
  yearsLabel: document.querySelector("#yearsLabel"),
  cityFilters: document.querySelector("#cityFilters"),
  skillFilters: document.querySelector("#skillFilters"),
  totalCount: document.querySelector("#totalCount"),
  queryMode: document.querySelector("#queryMode"),
  results: document.querySelector("#results"),
  drawer: document.querySelector("#drawer"),
  overlay: document.querySelector("#overlay"),
  drawerName: document.querySelector("#drawerName"),
  drawerMeta: document.querySelector("#drawerMeta"),
  drawerContent: document.querySelector("#drawerContent"),
  drawerClose: document.querySelector("#drawerClose"),
  clusterStatus: document.querySelector("#clusterStatus"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function params() {
  const query = new URLSearchParams();
  if (state.query) query.set("q", state.query);
  if (state.degree) query.set("degree", state.degree);
  if (state.minYears) query.set("min_years", state.minYears);
  state.cities.forEach((city) => query.append("cities", city));
  state.skills.forEach((skill) => query.append("skills", skill));
  return query.toString();
}

async function runSearch() {
  const seq = ++state.requestSeq;
  state.query = els.searchInput.value.trim();
  els.results.innerHTML = `<div class="empty-state">检索中...</div>`;
  const response = await fetch(`/api/search?${params()}`);
  const payload = await response.json();
  if (seq !== state.requestSeq) return;
  state.results = payload.results || [];
  els.totalCount.textContent = payload.returned_count ?? state.results.length;
  els.queryMode.textContent = state.query
    ? `搜索：${state.query}`
    : hasActiveFilters()
      ? "按筛选浏览"
      : "浏览全部候选人";
  syncQuickQueryState();
  renderFacets(payload.facets || {});
  renderResults();
}

function hasActiveFilters() {
  return Boolean(state.degree || state.minYears || state.cities.size || state.skills.size);
}

function formatYearsLabel(value) {
  return value > 0 ? `${value} 年以上` : "不限经验";
}

function syncQuickQueryState() {
  document.querySelectorAll(".quick-queries button").forEach((button) => {
    button.classList.toggle("active", button.dataset.query === state.query);
  });
}

function debounceSearch(delay = 300) {
  window.clearTimeout(state.debounceTimer);
  state.debounceTimer = window.setTimeout(() => {
    runSearch();
  }, delay);
}

function renderFacets(facets) {
  if (!els.cityFilters.dataset.ready) {
    renderChipGroup(els.cityFilters, facets.cities || [], state.cities, runSearch);
    els.cityFilters.dataset.ready = "1";
  }
  if (!els.skillFilters.dataset.ready) {
    renderChipGroup(els.skillFilters, facets.skills || [], state.skills, runSearch);
    els.skillFilters.dataset.ready = "1";
  }
}

function renderChipGroup(root, items, selectedSet, onChange) {
  root.innerHTML = "";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = "chip";
    button.type = "button";
    button.textContent = `${item.key} ${item.count}`;
    button.addEventListener("click", () => {
      if (selectedSet.has(item.key)) {
        selectedSet.delete(item.key);
        button.classList.remove("active");
      } else {
        selectedSet.add(item.key);
        button.classList.add("active");
      }
      onChange();
    });
    root.appendChild(button);
  });
}

const queryNameMap = {
  "candidate_no": "候选人编号",
  "position_code": "岗位编号",
  "candidate_name": "候选人姓名",
  "candidate_phone": "手机号",
  "candidate_email": "邮箱",
  "candidate_school": "学校",
  "candidate_major": "专业",
  "application_company": "应聘公司",
  "position_name": "岗位名称",
  "skills": "技能标签",
  "highest_degree": "最高学历",
  "wishes": "求职意向",
  "wish_company": "意向公司",
  "wish_position": "意向岗位",
  "education": "教育经历",
  "education_school": "教育-学校",
  "education_major": "教育-专业",
  "education_level": "教育-学历",
  "education_degree": "教育-学位",
  "education_college": "教育-学院",
  "education_research": "教育-研究方向",
  "education_lab": "教育-实验室",
  "internships": "实习经历",
  "internship_company": "实习-公司",
  "internship_work_type": "实习-性质",
  "internship_title": "实习-职位",
  "internship_department": "实习-部门",
  "internship_description": "实习-描述",
  "projects": "项目经历",
  "project_name": "项目-名称",
  "project_description": "项目-描述",
  "project_responsibility": "项目-职责",
  "section_education": "简历正文-教育",
  "section_projects": "简历正文-项目",
  "section_internships": "简历正文-实习"
};

function formatMatchedQuery(q) {
  if (q.startsWith("query_term:")) return null; // 隐藏由于覆盖率计算产生的底层 term 查询
  
  let type = "";
  let key = q;
  let typeClass = "";
  let weight = "";

  const wMatch = key.match(/:W([0-9.]+)$/);
  if (wMatch) {
    weight = wMatch[1];
    key = key.replace(wMatch[0], "");
  }

  if (key.startsWith("lexical_exact:")) {
    type = "[精确匹配]";
    typeClass = "exact";
    key = key.replace("lexical_exact:", "");
  } else if (key.startsWith("lexical_phrase:")) {
    type = "[短语匹配]";
    typeClass = "phrase";
    key = key.replace("lexical_phrase:", "");
  } else if (key.startsWith("lexical_term:")) {
    type = "[普通匹配]";
    typeClass = "term";
    key = key.replace("lexical_term:", "");
  }
  
  const fieldName = queryNameMap[key] || key;
  const text = weight ? `${type} ${fieldName} (权重:${weight})` : `${type} ${fieldName}`;
  return { typeClass, text, fieldName };
}

function renderResults() {
  if (!state.results.length) {
    els.results.innerHTML = `<div class="empty-state">没有匹配的候选人</div>`;
    return;
  }

  els.results.innerHTML = state.results
    .map((item, index) => {
      const candidate = item.candidate || {};
      const application = item.application || {};
      const skills = (item.skills || [])
        .slice(0, 6)
        .map((skill) => `<span class="skill-tag">${escapeHtml(skill)}</span>`)
        .join("");
        
      const debug = item.retrieval_debug || {};
      const sources = debug.retrieval_sources || [];
      const hasBm25 = sources.includes("bm25");
      const hasDense = sources.includes("dense");
      const rrfScore = debug.rrf_score ?? 0;

      const bm25Html = hasBm25 
        ? `<div class="debug-item"><span class="debug-label" title="Elasticsearch BM25 原生打分&#10;计算公式: ∑ (词频TF × 逆文档频率IDF × 字段权重Boost)&#10;此分数为下方命中详情中所有匹配特征的得分总和" style="cursor: help; text-decoration: underline dotted var(--muted); text-underline-offset: 4px;">BM25 排名：</span> <span class="debug-value">${debug.bm25_rank} <span class="tier-desc">(分数: ${debug.bm25_score || 0})</span></span></div>` 
        : `<div class="debug-item warning"><span class="debug-label">BM25 命中：</span> <span class="debug-value">否 (未召回)</span></div>`;
        
      const denseHtml = hasDense 
        ? `<div class="debug-item"><span class="debug-label">Dense 排名：</span> <span class="debug-value">${debug.dense_rank} <span class="tier-desc">(分数: ${debug.dense_score || 0})</span></span></div>` 
        : `<div class="debug-item warning"><span class="debug-label">Dense 命中：</span> <span class="debug-value">否 (未召回)</span></div>`;

      const rawQueries = (debug.matched_queries || [])
        .map(q => formatMatchedQuery(q))
        .filter(Boolean);
        
      const tierRank = { "exact": 3, "phrase": 2, "term": 1 };
      const uniqueMap = new Map();
      rawQueries.forEach(q => {
        const existing = uniqueMap.get(q.fieldName);
        if (!existing || tierRank[q.typeClass] > tierRank[existing.typeClass]) {
          uniqueMap.set(q.fieldName, q);
        }
      });
      
      const validQueries = Array.from(uniqueMap.values()).sort((a, b) => tierRank[b.typeClass] - tierRank[a.typeClass]);

      const matchedQueriesHtml = validQueries.length 
        ? `<div class="debug-item"><span class="debug-label">命中详情：</span>
             <div class="debug-queries">
               ${validQueries.map(q => `<span class="debug-tag tag-${q.typeClass}">${escapeHtml(q.text)}</span>`).join('')}
             </div>
           </div>`
        : '';

      const bm25Contrib = hasBm25 ? (1.5 / (60 + debug.bm25_rank)).toFixed(6) : "0";
      const denseContrib = hasDense ? (1 / (60 + debug.dense_rank)).toFixed(6) : "0";

      const debugPanelHtml = `
        <div class="debug-panel" style="display: none;">
          <div class="debug-header">
            <span class="debug-header-title">Debug 排名信息</span>
            <span class="debug-header-rank">最终排名 <strong class="highlight-number">${index + 1}</strong></span>
          </div>
          <div class="debug-content">
            <div class="debug-group status-group">
              <div class="debug-title">召回状态与分数</div>
              <div class="debug-list">
                ${bm25Html}
                ${denseHtml}
                <div class="debug-item"><span class="debug-label">双路均覆盖：</span> <span class="debug-value">${hasBm25 && hasDense ? '是' : '否'}</span></div>
              </div>
            </div>
            
            <div class="debug-group rrf-group">
              <div class="debug-title">RRF 融合分数计算过程</div>
              <div class="debug-formula-box">
                <div class="formula-row">
                  <span class="formula-label">BM25 贡献：</span>
                  <span class="formula-calc">${hasBm25 ? "1.5 / (60 + " + debug.bm25_rank + ")" : '0 (未命中)'} <span class="formula-eq">=</span> <span class="formula-val">${bm25Contrib}</span></span>
                </div>
                <div class="formula-row">
                  <span class="formula-label">Dense 贡献：</span>
                  <span class="formula-calc">${hasDense ? "1 / (60 + " + debug.dense_rank + ")" : '0 (未命中)'} <span class="formula-eq">=</span> <span class="formula-val">${denseContrib}</span></span>
                </div>
                <div class="formula-divider"></div>
                <div class="formula-row highlight-row">
                  <span class="formula-label">基础 RRF 分数：</span>
                  <span class="formula-calc"><span class="formula-val primary-val">${debug.raw_rrf_score !== undefined ? debug.raw_rrf_score : rrfScore}</span></span>
                </div>
                ${debug.score_multiplier !== undefined ? `
                <div class="formula-row" style="margin-bottom: 4px;">
                  <span class="formula-label">层级与覆盖度奖励系数：</span>
                  <span class="formula-calc"><span class="formula-val" style="color: var(--text);">x${debug.score_multiplier.toFixed(2)}</span></span>
                </div>
                <div style="text-align: right; color: #94a3b8; font-size: 12px; margin-bottom: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; letter-spacing: 0.5px;">
                  1.0(基础) + 0.15×${debug.lexical_tier || 0}(层级) + 0.05×${debug.term_coverage || 0}(覆盖数)
                </div>
                <div class="formula-divider strong"></div>
                <div class="formula-row final-row">
                  <span class="formula-label">最终加权得分：</span>
                  <span class="formula-calc"><span class="formula-val final-val">${rrfScore}</span></span>
                </div>` : ''}
              </div>
            </div>

            ${debug.matched_queries && debug.matched_queries.length ? `
            <div class="debug-group queries-group">
              <div class="debug-title">关键词与字段命中详情</div>
              <div class="debug-list">
                ${debug.term_coverage !== undefined ? `<div class="debug-item"><span class="debug-label">Term 覆盖：</span> <span class="debug-value">${debug.term_coverage}</span></div>` : ''}
                <div class="debug-item"><span class="debug-label">匹配层级：</span> <span class="debug-value tier-${debug.lexical_tier || 0}">${debug.lexical_tier || 0} <span class="tier-desc">(${debug.lexical_tier === 3 ? '完全匹配' : debug.lexical_tier === 2 ? '短语部分匹配' : '普通匹配'})</span></span></div>
                ${matchedQueriesHtml}
              </div>
            </div>` : ''}
          </div>
        </div>
      `;

      return `
        <article class="result-card" data-id="${escapeHtml(item.id)}">
          <div class="result-main">
            <div class="candidate-line">
              <div class="candidate-name">${escapeHtml(candidate.name || "未命名候选人")}</div>
            </div>
            <div class="result-subtitle">${escapeHtml(application.position_name || "岗位未标注")} · ${escapeHtml(application.position_code || "")}</div>
            <div class="meta-line">${buildMetaLineHTML(item)}</div>
            <div class="snippet">${item.project_snippet}</div>
            <div class="skill-row">${skills || `<span class="skill-tag">技能待补充</span>`}</div>
          </div>
          <div class="result-actions">
            ${sources.length > 0 ? `<button class="quick-action debug-action" type="button" style="margin-right: 8px;">Debug 排名</button>` : ''}
            <button class="quick-action details-action" type="button">查看详情</button>
          </div>
          ${sources.length > 0 ? debugPanelHtml : ''}
        </article>
      `;
    })
    .join("");

  document.querySelectorAll(".result-card").forEach((card) => {
    card.querySelector(".candidate-name")?.addEventListener("click", () => {
      const item = state.results.find((result) => result.id === card.dataset.id);
      if (item) openDrawer(item);
    });
    card.querySelector(".details-action")?.addEventListener("click", () => {
      const item = state.results.find((result) => result.id === card.dataset.id);
      if (item) openDrawer(item);
    });
    card.querySelector(".debug-action")?.addEventListener("click", (e) => {
      const debugPanel = card.querySelector(".debug-panel");
      if (debugPanel.style.display === "none") {
        debugPanel.style.display = "block";
        e.target.textContent = "收起 Debug";
      } else {
        debugPanel.style.display = "none";
        e.target.textContent = "Debug 排名";
      }
    });
  });
}

function openDrawer(item) {
  const source = item.source || {};
  const candidate = source.candidate || {};
  const application = source.application || {};
  els.drawerName.textContent = candidate.name || "未命名候选人";
  els.drawerMeta.textContent = `${application.position_name || "岗位未标注"} · ${application.candidate_no || ""}`;
  els.drawerContent.innerHTML = renderDetails(source);
  els.overlay.classList.add("open");
  els.drawer.classList.add("open");
}

function closeDrawer() {
  els.overlay.classList.remove("open");
  els.drawer.classList.remove("open");
}

function renderDetails(source) {
  const candidate = source.candidate || {};
  const application = source.application || {};
  const educations = source.education || [];
  const internships = source.internships || [];
  const projects = source.projects || [];
  const languages = source.languages || {};
  const skills = source.skills || [];
  const itSkillItems = source.it_skill_items || [];
  const awards = source.awards || [];
  const offerInternship = source.offer_internship || {};
  const uploadedResume = source.uploaded_resume || {};

  return `
    <section class="detail-section">
      <h3>基础信息</h3>
      <div class="detail-grid">
        ${detail("姓名", candidate.name)}
        ${detail("性别", candidate.gender)}
        ${detail("出生日期", candidate.birth_date)}
        ${detail("民族", candidate.ethnicity)}
        ${detail("国籍", candidate.nationality)}
        ${detail("政治面貌", candidate.political_status)}
        ${detail("证件信息", candidate.identity_info)}
        ${detail("手机", candidate.phone)}
        ${detail("紧急联系电话", candidate.emergency_phone)}
        ${detail("邮箱", candidate.email)}
        ${detail("当前城市", candidate.current_city)}
        ${detail("入学前户口所在城市", candidate.pre_college_residence_city)}
        ${detail("最高学位", candidate.highest_degree)}
        ${detail("毕业院校", candidate.school)}
        ${detail("专业", candidate.major)}
        ${detail("毕业时间", candidate.graduation_date)}
        ${detail("是否接受调剂", candidate.accept_transfer)}
        ${detail("可面试城市", candidate.interview_city)}
        ${detail("招聘信息来源", candidate.recruiting_source)}
        ${detail("内推码", candidate.referral_code)}
      </div>
    </section>
    <section class="detail-section">
      <h3>应聘信息</h3>
      <div class="detail-grid">
        ${detail("候选人编号", application.candidate_no)}
        ${detail("投递时间", application.apply_time)}
        ${detail("应聘公司", application.company)}
        ${detail("应聘岗位", application.position_name)}
        ${detail("岗位编号", application.position_code)}
        ${detail("期望工作城市", (application.expected_work_cities || []).join("、"))}
      </div>
    </section>
    ${timeline("教育经历", educations, (item) => ({
      title: [item.school, item.degree || item.education_level, item.major].filter(Boolean).join(" / "),
      meta: `${item.start_date_raw || item.start_date || ""} - ${item.end_date_raw || item.end_date || ""}`,
      body: [
        item.college ? "学院: " + item.college : "",
        item.research_direction ? "研究方向: " + item.research_direction : "",
        item.lab_name ? "实验室: " + item.lab_name : "",
        item.advisor_name ? "导师: " + item.advisor_name : "",
        item.advisor_contact ? "导师联系方式: " + item.advisor_contact : "",
        item.courses ? "主修课程: " + item.courses : "",
        item.paper_level ? "论文发表等级: " + item.paper_level : "",
        item.github ? "GitHub: " + item.github : "",
      ].filter(Boolean).join(" · "),
    }))}
    ${timeline("实习经历", internships, (item) => ({
      title: [item.company, item.title].filter(Boolean).join(" / "),
      meta: `${item.start_date_raw || item.start_date || ""} - ${item.end_date_raw || item.end_date || ""}`,
      body: [
        item.department ? "部门: " + item.department : "",
        item.work_type ? "工作性质: " + item.work_type : "",
        item.company_type ? "企业性质: " + item.company_type : "",
        item.company_size ? "企业规模: " + item.company_size : "",
        item.description,
      ].filter(Boolean).join(" · "),
    }))}
    ${timeline("项目经验", projects, (item) => ({
      title: item.name,
      meta: `${item.start_date_raw || item.start_date || ""} - ${item.end_date_raw || item.end_date || ""}`,
      body: [
        item.description,
        item.responsibility ? "项目职责: " + item.responsibility : "",
      ].filter(Boolean).join(" "),
    }))}
    ${timeline("奖项与活动", awards, (item) => ({
      title: item.name || "未命名奖项",
      meta: item.level || "",
      body: item.description || (item.has_award ? "有获奖经历" : ""),
    }))}
    <section class="detail-section">
      <h3>IT 技能</h3>
      ${itSkillItems.length ? itSkillItems.map((item) => `
        <div class="timeline-item">
          <div class="timeline-title">${escapeHtml(item.skill_name || "未命名技能")}</div>
          <div class="timeline-meta">${escapeHtml([item.duration, item.proficiency].filter(Boolean).join(" · "))}</div>
          <div class="timeline-body">${escapeHtml([item.primary_languages ? "主要语言: " + item.primary_languages : "", item.other_languages ? "其它语言: " + item.other_languages : ""].filter(Boolean).join(" · ") || "暂无描述")}</div>
        </div>
      `).join("") : `<div class="timeline-item"><div class="timeline-body">暂无记录</div></div>`}
    </section>
    <section class="detail-section">
      <h3>语言能力</h3>
      <div class="detail-grid">
        ${detail("英语等级考试成绩", languages.english_exam_score)}
        ${detail("英语口语水平", languages.english_spoken_level)}
      </div>
    </section>
    <section class="detail-section">
      <h3>实习与入职意向</h3>
      <div class="detail-grid">
        ${detail("毕业后意向", offerInternship.post_graduation_intention)}
        ${detail("是否可以实习", offerInternship.can_intern)}
        ${detail("可开始工作日期", offerInternship.available_start_date)}
        ${detail("每周可实习天数", offerInternship.weekly_workdays)}
        ${detail("可实习周期", offerInternship.internship_period)}
      </div>
    </section>
    <section class="detail-section">
      <h3>上传简历</h3>
      <div class="detail-grid">
        ${detail("中文简历", uploadedResume.chinese_resume)}
      </div>
    </section>
    ${skills.length ? `
    <section class="detail-section">
      <h3>技能标签</h3>
      <div class="detail-grid">
        <div class="detail-item" style="grid-column: 1 / -1;">
          <div class="detail-label">技能列表</div>
          <div class="detail-value">${escapeHtml(skills.join("、"))}</div>
        </div>
      </div>
    </section>
    ` : ""}
  `;
}

function detail(label, value) {
  return `
    <div class="detail-item">
      <div class="detail-label">${escapeHtml(label)}</div>
      <div class="detail-value">${escapeHtml(value || "待补充")}</div>
    </div>
  `;
}

function timeline(title, items, mapper) {
  const body = items.length
    ? items
        .map((item) => {
          const row = mapper(item);
          return `
            <div class="timeline-item">
              <div class="timeline-title">${escapeHtml(row.title || "未命名")}</div>
              <div class="timeline-meta">${escapeHtml(row.meta || "")}</div>
              <div class="timeline-body">${escapeHtml(row.body || "暂无描述")}</div>
            </div>
          `;
        })
        .join("")
    : `<div class="timeline-item"><div class="timeline-body">暂无记录</div></div>`;
  return `<section class="detail-section"><h3>${escapeHtml(title)}</h3>${body}</section>`;
}

els.searchButton.addEventListener("click", runSearch);
els.searchInput.addEventListener("input", () => {
  debounceSearch();
});
els.searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    window.clearTimeout(state.debounceTimer);
    runSearch();
  }
});
els.yearsRange.addEventListener("input", () => {
  state.minYears = Number(els.yearsRange.value);
  els.yearsLabel.textContent = formatYearsLabel(state.minYears);
});
els.yearsRange.addEventListener("change", runSearch);
document.querySelectorAll('input[name="degree"]').forEach((input) => {
  input.addEventListener("change", () => {
    state.degree = input.value;
    runSearch();
  });
});
document.querySelectorAll(".quick-queries button").forEach((button) => {
  button.addEventListener("click", () => {
    els.searchInput.value = button.dataset.query;
    els.searchInput.focus();
    window.clearTimeout(state.debounceTimer);
    runSearch();
  });
});
els.drawerClose.addEventListener("click", closeDrawer);
els.overlay.addEventListener("click", closeDrawer);

function buildMetaLineHTML(item) {
  const source = item.source || {};
  const education = source.education || [];
  const yearsDisplay = item.experience_display || "";

  const parts = [];

  // 教育经历
  if (education.length > 0) {
    const eduHTML = education
      .map((edu) => {
        const school = edu.school || "";
        const degree = edu.degree || edu.education_level || "";
        const major = edu.major || "";
        const core = [school, degree].filter(Boolean).join("");
        const label = major ? `${core} ${major}` : core;
        return label ? `<span class="meta-edu">${escapeHtml(label)}</span>` : "";
      })
      .filter(Boolean)
      .join('<span class="meta-sep"></span>');
    if (eduHTML) parts.push(eduHTML);
  }

  // 工作经验
  if (yearsDisplay) {
    parts.push(
      `<span class="meta-exp">${escapeHtml(yearsDisplay)}</span>`
    );
  }

  return parts.length
    ? parts.join('<span class="meta-divider"></span>')
    : "信息待补充";
}

async function checkESHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const statusEl = els.clusterStatus;
    if (data.es_online) {
      statusEl.className = "cluster-status online";
      statusEl.querySelector("em").textContent = "在线";
    } else {
      statusEl.className = "cluster-status offline";
      statusEl.querySelector("em").textContent = "离线";
    }
  } catch {
    els.clusterStatus.className = "cluster-status offline";
    els.clusterStatus.querySelector("em").textContent = "离线";
  }
}

checkESHealth();
setInterval(checkESHealth, 30000);

runSearch();
