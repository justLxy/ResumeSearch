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
  if (facets.max_years != null) {
    const maxYears = Math.floor(facets.max_years);
    if (maxYears > 0 && els.yearsRange.max !== String(maxYears)) {
      els.yearsRange.max = maxYears;
    }
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
  "evidence_title": "证据-标题",
  "evidence_text": "证据-正文",
  "evidence_candidate_no": "档案-候选人编号",
  "evidence_position_code": "档案-岗位编号",
  "evidence_candidate_name": "档案-姓名",
  "evidence_candidate_phone": "档案-手机",
  "evidence_candidate_email": "档案-邮箱",
  "evidence_candidate_school": "档案-学校",
  "evidence_candidate_major": "档案-专业",
  "evidence_application_company": "档案-公司",
  "evidence_highest_degree": "档案-学历",
  "evidence_skills": "档案-技能",
  "evidence_position_name": "证据-岗位",
  "evidence_all_terms": "证据-全部词命中",
  "evidence_partial_terms": "证据-部分词命中",
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
  } else if (key.startsWith("evidence_exact:")) {
    type = "[证据精确]";
    typeClass = "exact";
    key = `evidence_${key.replace("evidence_exact:", "")}`;
  } else if (key.startsWith("evidence_phrase:")) {
    type = "[证据短语]";
    typeClass = "phrase";
    key = `evidence_${key.replace("evidence_phrase:", "")}`;
  } else if (key.startsWith("evidence_term:")) {
    type = "[证据词面]";
    typeClass = "term";
    key = `evidence_${key.replace("evidence_term:", "")}`;
  }
  
  const fieldName = queryNameMap[key] || key;
  const text = weight ? `${type} ${fieldName} (权重:${weight})` : `${type} ${fieldName}`;
  return { typeClass, text, fieldName };
}

const denseFieldNameMap = {
  evidence_vector: "证据向量",
  skills_vector: "技能向量",
  projects_vector: "项目向量",
  internships_vector: "实习向量",
  education_vector: "教育向量",
};

const evidenceSectionNameMap = {
  profile: "候选人档案",
  skills: "技能标签",
  project: "项目经历",
  internship: "实习经历",
  education: "教育经历",
};

function isDenseSource(source) {
  return source === "dense" || String(source || "").startsWith("dense:");
}

function denseFieldLabel(field, retriever) {
  if (field && denseFieldNameMap[field]) return denseFieldNameMap[field];
  const name = String(retriever || "").replace("dense:", "");
  if (name && name !== retriever) return `${name} 向量`;
  return "Dense 向量";
}

function evidenceSectionLabel(sectionType) {
  return evidenceSectionNameMap[sectionType] || "证据片段";
}

function denseMatchLabel(match) {
  if (match?.section_type) {
    const section = evidenceSectionLabel(match.section_type);
    const title = String(match.title || "").trim();
    return title && title !== section ? `${section}｜${title}` : section;
  }
  return denseFieldLabel(match?.field, match?.retriever);
}

function formatScore(value, digits = 6) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "0";
}

function denseContribution(match) {
  const contribution = Number(match.contribution);
  if (Number.isFinite(contribution)) return contribution;
  const weight = Number(match.weight ?? 1);
  const rank = Number(match.rank);
  if (!Number.isFinite(rank) || rank <= 0) return 0;
  return weight / (60 + rank);
}

function denseOuterContribution(debug) {
  const contribution = Number(debug.dense_rrf_contribution);
  if (Number.isFinite(contribution)) return contribution;
  const weight = Number(debug.dense_outer_weight ?? 1);
  const rank = Number(debug.dense_group_rank ?? debug.dense_rank);
  if (!Number.isFinite(rank) || rank <= 0) return 0;
  return weight / (60 + rank);
}

function denseMatchesForDebug(debug) {
  const matches = Array.isArray(debug.dense_matches) ? debug.dense_matches : [];
  if (matches.length) {
    return matches
      .filter((match) => match && match.rank)
      .sort((a, b) => denseContribution(b) - denseContribution(a));
  }
  if (debug.dense_rank) {
    return [{
      retriever: "dense",
      field: debug.dense_field,
      rank: debug.dense_rank,
      score: debug.dense_score,
      weight: 1,
    }];
  }
  return [];
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
      const hasEvidenceLexical = sources.includes("evidence") && debug.evidence_rank;
      const hasLexical = hasEvidenceLexical;
      const hasDense = sources.some(isDenseSource);
      const rrfScore = debug.rrf_score ?? 0;
      const denseMatches = denseMatchesForDebug(debug);
      const denseOuterWeight = Number(debug.dense_outer_weight ?? 1);
      const denseGroupRank = debug.dense_group_rank ?? debug.dense_rank;
      const bestDenseRouteRank = debug.dense_route_rank ?? denseMatches[0]?.rank ?? debug.dense_rank;
      const evidenceGroupRank = debug.evidence_group_rank ?? debug.evidence_rank;
      const bestDenseLabel = denseMatches[0] ? denseMatchLabel(denseMatches[0]) : denseFieldLabel(debug.dense_field, null);

      const lexicalStatusRows = [
        hasEvidenceLexical
          ? `<div class="debug-item"><span class="debug-label" title="证据索引中的 BM25 / phrase 词面召回&#10;先命中候选人档案、项目、实习、教育或技能证据片段，再按候选人聚合排名" style="cursor: help; text-decoration: underline dotted var(--muted); text-underline-offset: 4px;">词面证据排名：</span> <span class="debug-value">${evidenceGroupRank} <span class="tier-desc">(最佳片段 #${debug.evidence_rank}，分数: ${formatScore(debug.evidence_score || 0, 4)})</span></span></div>`
          : "",
      ].filter(Boolean).join("");
      const lexicalHtml = hasLexical
        ? lexicalStatusRows
        : `<div class="debug-item warning"><span class="debug-label">词面证据命中：</span> <span class="debug-value">否 (未召回)</span></div>`;
        
      const denseHtml = hasDense 
        ? `<div class="debug-item"><span class="debug-label" title="先在 evidence_vector 中检索证据片段，再按 resume_id 在完整向量结果中聚合成候选人排名&#10;括号里的 # 是最佳证据片段在原始 kNN 结果里的名次">全局向量聚合排名：</span> <span class="debug-value">${denseGroupRank} <span class="tier-desc">(最佳证据 ${escapeHtml(bestDenseLabel)} 原始 #${bestDenseRouteRank}，分数: ${formatScore(debug.dense_score || 0, 4)})</span></span></div>`
        : `<div class="debug-item warning"><span class="debug-label">Dense 命中：</span> <span class="debug-value">否 (未召回)</span></div>`;

      const denseMatchesHtml = denseMatches.length
        ? `<div class="debug-item dense-match-row"><span class="debug-label">向量详情：</span>
             <div class="dense-match-list">
               ${denseMatches.map((match) => `
                 <div class="dense-match-item">
                   <div class="dense-match-main">
                     <span class="dense-match-name">${escapeHtml(denseMatchLabel(match))} #${escapeHtml(match.rank)}</span>
                     <span class="dense-match-score">分数 ${escapeHtml(formatScore(match.score, 4))}</span>
                   </div>
                   ${match.snippet ? `<div class="dense-match-snippet">${match.snippet}</div>` : ""}
                 </div>
               `).join("")}
             </div>
           </div>`
        : "";

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

      const evidenceLexicalContrib = hasEvidenceLexical
        ? (debug.evidence_rrf_contribution != null
            ? formatScore(debug.evidence_rrf_contribution, 6)
            : (Number(debug.evidence_weight ?? 1.2) / (60 + Number(evidenceGroupRank))).toFixed(6))
        : "0";
      const lexicalFormulaRowsHtml = `
        <div class="formula-row">
          <span class="formula-label">词面证据贡献：</span>
          <span class="formula-calc">${hasEvidenceLexical ? formatScore(debug.evidence_weight ?? 1.2, 2) + " / (60 + " + evidenceGroupRank + ")" : '0 (未命中)'} <span class="formula-eq">=</span> <span class="formula-val">${evidenceLexicalContrib}</span></span>
        </div>`;
      const denseContrib = hasDense ? denseOuterContribution(debug).toFixed(6) : "0";
      const denseFormulaHtml = hasDense
        ? `<div class="formula-row">
             <span class="formula-label">Dense 贡献：</span>
             <span class="formula-calc">${escapeHtml(formatScore(denseOuterWeight, 2))} / (60 + ${escapeHtml(denseGroupRank)}) <span class="formula-eq">=</span> <span class="formula-val">${denseContrib}</span></span>
           </div>`
        : `<div class="formula-row">
             <span class="formula-label">Dense 贡献：</span>
             <span class="formula-calc">0 (未命中) <span class="formula-eq">=</span> <span class="formula-val">0</span></span>
           </div>`;

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
                ${lexicalHtml}
                ${denseHtml}
                ${denseMatchesHtml}
              </div>
            </div>
            
            <div class="debug-group rrf-group">
              <div class="debug-title">RRF 融合分数计算过程</div>
              <div class="debug-formula-box">
                ${lexicalFormulaRowsHtml}
                ${denseFormulaHtml}
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
      attrs: [
        item.college ? ["学院", item.college] : null,
        item.research_direction ? ["研究方向", item.research_direction] : null,
        item.lab_name ? ["实验室", item.lab_name] : null,
        item.advisor_name ? ["导师", item.advisor_name] : null,
        item.advisor_contact ? ["导师联系方式", item.advisor_contact] : null,
        item.paper_level ? ["论文发表等级", item.paper_level] : null,
      ].filter(Boolean),
      desc: item.courses ? "主修课程: " + item.courses : "",
    }))}
    ${timeline("实习经历", internships, (item) => ({
      title: [item.company, item.title].filter(Boolean).join(" / "),
      meta: `${item.start_date_raw || item.start_date || ""} - ${item.end_date_raw || item.end_date || ""}`,
      attrs: [
        item.department ? ["部门", item.department] : null,
        item.work_type ? ["工作性质", item.work_type] : null,
        item.company_type ? ["企业性质", item.company_type] : null,
        item.company_size ? ["企业规模", item.company_size] : null,
      ].filter(Boolean),
      desc: item.description || "",
    }))}
    ${timeline("项目经验", projects, (item) => ({
      title: item.name,
      meta: `${item.start_date_raw || item.start_date || ""} - ${item.end_date_raw || item.end_date || ""}`,
      attrs: [
        item.project_type ? ["项目类型", item.project_type] : null,
      ].filter(Boolean),
      desc: [item.description, item.responsibility ? "项目职责: " + item.responsibility : ""].filter(Boolean).join("\n"),
    }))}
    ${timeline("奖项与活动", awards, (item) => ({
      title: item.name || "未命名奖项",
      meta: item.level || "",
      body: item.description || (item.has_award ? "有获奖经历" : ""),
    }))}
    <section class="detail-section">
      <h3>IT 技能</h3>
      ${(() => {
        if (!itSkillItems.length) return '<div class="timeline-body">暂无记录</div>';
        const first = itSkillItems[0] || {};
        const langParts = [
          first.primary_languages ? "主要语言: " + first.primary_languages : "",
          first.other_languages ? "其它语言: " + first.other_languages : "",
        ].filter(Boolean);
        const langHtml = langParts.length
          ? `<div class="it-skill-langs">${escapeHtml(langParts.join(" · "))}</div>`
          : "";
        const skillsHtml = `<div class="it-skill-grid">${itSkillItems.map((item) => {
          const meta = [item.duration, item.proficiency].filter(Boolean).join(" · ");
          return `<span class="it-skill-chip"><span class="it-skill-name">${escapeHtml(item.skill_name || "未命名")}</span>${meta ? `<span class="it-skill-meta">${escapeHtml(meta)}</span>` : ""}</span>`;
        }).join("")}</div>`;
        return langHtml + skillsHtml;
      })()}
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
          const attrsHtml = (row.attrs || []).length
            ? `<div class="timeline-attrs">${row.attrs.map(([label, value]) =>
                `<span class="timeline-attr"><span class="timeline-attr-label">${escapeHtml(label)}</span>${escapeHtml(value)}</span>`
              ).join("")}</div>`
            : "";
          const descHtml = row.desc
            ? `<div class="timeline-body">${escapeHtml(row.desc)}</div>`
            : "";
          // Fallback: support legacy plain-string body
          const legacyBodyHtml = (!row.attrs && !row.desc && row.body)
            ? `<div class="timeline-body">${escapeHtml(row.body)}</div>`
            : "";
          const hasContent = attrsHtml || descHtml || legacyBodyHtml;
          return `
            <div class="timeline-item">
              <div class="timeline-title">${escapeHtml(row.title || "未命名")}</div>
              <div class="timeline-meta">${escapeHtml(row.meta || "")}</div>
              ${hasContent ? (attrsHtml + descHtml + legacyBodyHtml) : '<div class="timeline-body">暂无描述</div>'}
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
