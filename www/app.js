let allItems = [];
let items = [];
let idx = 0;
let allTags = [];
let tagify;
let saving = false;
let filterMode = false;
let galleryIndex = -1;
let selectedCluster = '';
let lastMlStage = '';
let selectedPaths = new Set();
let visibleGalleryItems = [];
let selectionAnchorIndex = -1;
let lastMlStatusKey = '';
let mlPollDelayMs = 3000;
let mlPollTimer = null;
let forceSingleView = false;
let galleryExpanded = false;

const shot = document.getElementById('shot');
const progress = document.getElementById('progress');
const filepath = document.getElementById('filepath');
const tagbar = document.getElementById('tagbar');
const jump = document.getElementById('jump');
const total = document.getElementById('total');
const filterToggle = document.getElementById('filter-toggle');
const labelsDropdown = document.getElementById('labels');
const clustersDropdown = document.getElementById('clusters');
const gallery = document.getElementById('gallery');
const mlStatus = document.getElementById('ml-status');
const mlSources = document.getElementById('ml-sources');
const bulkbar = document.getElementById('bulkbar');
const bulkCount = document.getElementById('bulk-count');
const bulkSelectToggle = document.getElementById('bulk-select-toggle');
const bulkApply = document.getElementById('bulk-apply');
const bulkExpand = document.getElementById('bulk-expand');

async function fetchJson(url, opts = undefined) {
  const response = await fetch(url, opts);
  return response.json();
}

function labeledCount() {
  return allItems.filter(item => (item.categories || []).length > 0).length;
}

function clearSelection() {
  selectedPaths.clear();
  selectionAnchorIndex = -1;
}

function updateBulkBar(selectionModeEnabled, galleryEnabled) {
  if (!selectionModeEnabled) {
    bulkbar.classList.add('hidden');
    bulkCount.textContent = '0 selected';
    return;
  }
  bulkbar.classList.remove('hidden');
  const n = selectedPaths.size;
  bulkCount.textContent = `${n} selected`;
  bulkApply.disabled = n === 0;
  const visibleCount = visibleGalleryItems.length;
  const selectedVisible = visibleGalleryItems.filter(item => selectedPaths.has(item.input_path)).length;
  const allVisibleSelected = visibleCount > 0 && selectedVisible === visibleCount;
  bulkSelectToggle.textContent = allVisibleSelected ? 'select none' : 'select all';
  bulkSelectToggle.disabled = visibleCount === 0;
  if (!galleryEnabled) {
    bulkExpand.textContent = 'expand';
  } else {
    bulkExpand.textContent = galleryExpanded ? 'collapse' : 'expand';
  }
}

function applyClusterFilter(preservePath = true) {
  const currentPath = preservePath ? items[idx]?.input_path : '';
  clearSelection();
  forceSingleView = false;
  if (!selectedCluster) {
    items = [...allItems];
  } else {
    items = allItems.filter(item => String(item.cluster ?? '') === selectedCluster);
  }
  if (!items.length) {
    idx = 0;
    render();
    return;
  }
  if (currentPath) {
    const next = items.findIndex(item => item.input_path === currentPath);
    idx = next >= 0 ? next : Math.min(idx, items.length - 1);
  } else {
    idx = Math.min(idx, items.length - 1);
  }
  render();
}

async function reloadItems() {
  allItems = await fetchJson('/api/items');
  applyClusterFilter(true);
}

async function applyTagsToSelected() {
  if (saving || selectedPaths.size === 0) return;
  const tags = tagify.value.map(entry => entry.value);
  const targets = allItems.filter(item => selectedPaths.has(item.input_path));
  if (!targets.length) return;

  saving = true;
  try {
    tags.forEach(tag => {
      if (!allTags.includes(tag)) allTags.push(tag);
    });
    tagify.settings.whitelist = [...allTags];

    targets.forEach(item => {
      item.categories = [...tags];
    });
    items.forEach(item => {
      if (selectedPaths.has(item.input_path)) {
        item.categories = [...tags];
      }
    });

    const responses = await Promise.all(
      targets.map(item => fetch(`/api/item/${item._idx}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ categories: tags }),
      })),
    );
    if (responses.some(response => !response.ok)) {
      throw new Error('bulk tag apply failed');
    }
    progress.textContent = `${labeledCount()} labeled`;
    renderLabelsDropdown();
    clearSelection();
    renderGallery();
  } finally {
    saving = false;
  }
}

function renderClusterDropdown(clusters) {
  const current = selectedCluster;
  const options = ['<option value="">all clusters</option>'].concat(
    (clusters || []).map(cluster => (
      `<option value="${cluster.id}">c${cluster.id} (${cluster.count})</option>`
    )),
  );
  clustersDropdown.innerHTML = options.join('');
  clustersDropdown.value = current;
}

async function refreshClusters() {
  const clusters = await fetchJson('/api/ml/clusters');
  renderClusterDropdown(clusters);
}

function formatMlStatus(status) {
  const stage = status.stage || 'idle';
  if (stage === 'embedding') {
    const done = status.processed_images || 0;
    const count = status.total_images || 0;
    const rate = status.rate_images_per_second || 0;
    const eta = status.eta_seconds || 0;
    return `ml: embedding ${done}/${count} · ${rate.toFixed(2)} img/s · eta ${eta}s`;
  }
  if (stage === 'clustering') {
    return `ml: clustering k=${status.cluster_count || '—'}`;
  }
  if (stage === 'done') {
    const count = status.total_images || 0;
    const k = status.cluster_count || '—';
    return `ml: done ${count} images · k=${k}`;
  }
  if (stage === 'error') {
    return `ml: error ${status.error || ''}`.trim();
  }
  if (stage === 'scanning') {
    const count = status.total_images || 0;
    return `ml: scanning sources (${count} images)`;
  }
  return `ml: ${stage}`;
}

function formatSourceStats(status) {
  const stats = status.source_stats || [];
  if (!stats.length) {
    const roots = status.source_roots || [];
    if (!roots.length) return '';
    return roots.join(' | ');
  }
  return stats
    .map(row => `${row.series}/${row.source}: ${row.count}`)
    .join(' | ');
}

async function refreshMlStatus() {
  const status = await fetchJson('/api/ml/status');
  const stage = status.stage || 'idle';
  const statusKey = [
    stage,
    status.processed_images || 0,
    status.total_images || 0,
    status.cluster_count || 0,
    status.eta_seconds || 0,
    status.error || '',
  ].join(':');
  const changed = statusKey !== lastMlStatusKey;
  mlStatus.textContent = formatMlStatus(status);
  mlSources.textContent = formatSourceStats(status);
  if (stage === 'done' && lastMlStage !== 'done') {
    await Promise.all([reloadItems(), refreshClusters()]);
  }
  lastMlStage = stage;
  lastMlStatusKey = statusKey;

  if (stage === 'embedding' || stage === 'clustering' || stage === 'scanning') {
    mlPollDelayMs = changed ? 3000 : Math.min(Math.round(mlPollDelayMs * 1.5), 15000);
    return;
  }
  if (stage === 'done' || stage === 'error') {
    mlPollDelayMs = changed ? 10000 : Math.min(Math.round(mlPollDelayMs * 2), 120000);
    return;
  }
  mlPollDelayMs = changed ? 5000 : Math.min(Math.round(mlPollDelayMs * 1.5), 60000);
}

async function pollMlStatus() {
  await refreshMlStatus();
  mlPollTimer = setTimeout(pollMlStatus, mlPollDelayMs);
}

async function initMl() {
  await fetchJson('/api/ml/start', { method: 'POST' });
  await Promise.all([refreshMlStatus(), refreshClusters()]);
  if (mlPollTimer !== null) {
    clearTimeout(mlPollTimer);
  }
  mlPollTimer = setTimeout(pollMlStatus, mlPollDelayMs);
}

async function init() {
  const [all, tags] = await Promise.all([
    fetchJson('/api/items'),
    fetchJson('/api/tags'),
  ]);
  allItems = all;
  allTags = tags;
  items = [...allItems];

  tagify = new Tagify(document.getElementById('tag-input'), {
    whitelist: [...allTags],
    dropdown: { enabled: 2, closeOnSelect: false, maxItems: 30 },
  });
  tagify.on('add', onTagChange);
  tagify.on('remove', onTagChange);

  jump.addEventListener('click', () => {
    jump.contentEditable = 'true';
    jump.classList.add('editing');
    jump.focus();
    document.execCommand('selectAll', false, null);
  });

  jump.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      const target = Number.parseInt(jump.textContent, 10);
      if (!Number.isNaN(target)) {
        idx = Math.max(0, Math.min(items.length - 1, target - 1));
        render();
      }
      jump.blur();
      event.preventDefault();
    }
    if (event.key === 'Escape') {
      jump.textContent = String(idx + 1);
      jump.blur();
      event.preventDefault();
    }
  });

  jump.addEventListener('blur', () => {
    jump.contentEditable = 'false';
    jump.classList.remove('editing');
  });

  filterToggle.addEventListener('click', () => {
    filterMode = !filterMode;
    filterToggle.classList.toggle('active', filterMode);
    if (!filterMode && !selectedCluster) clearSelection();
    forceSingleView = false;
    render();
  });

  labelsDropdown.addEventListener('change', () => {
    const selected = labelsDropdown.value;
    if (!selected) return;
    tagify.addTags([selected]);
    labelsDropdown.value = '';
  });

  clustersDropdown.addEventListener('change', () => {
    selectedCluster = clustersDropdown.value;
    applyClusterFilter(true);
  });

  bulkSelectToggle.addEventListener('click', () => {
    const visible = visibleGalleryItems.map(item => item.input_path);
    const selectedVisible = visible.filter(path => selectedPaths.has(path));
    const allVisibleSelected = visible.length > 0 && selectedVisible.length === visible.length;
    if (allVisibleSelected) {
      visible.forEach(path => selectedPaths.delete(path));
    } else {
      visible.forEach(path => selectedPaths.add(path));
    }
    renderGallery();
  });

  bulkApply.addEventListener('click', () => {
    applyTagsToSelected().catch(error => {
      console.error(error);
    });
  });

  bulkExpand.addEventListener('click', () => {
    if (forceSingleView) {
      forceSingleView = false;
      render();
      return;
    }
    galleryExpanded = !galleryExpanded;
    renderGallery();
  });

  const saved = Number.parseInt(localStorage.getItem('tagger-index') || '', 10);
  idx = Number.isNaN(saved) ? 0 : Math.max(0, Math.min(items.length - 1, saved));
  render();

  await initMl();
}

function render() {
  total.textContent = String(items.length);
  progress.textContent = `${labeledCount()} labeled`;

  const item = items[idx];
  if (!item) {
    shot.src = '';
    filepath.textContent = selectedCluster
      ? `no images in cluster c${selectedCluster}`
      : 'no images';
    jump.textContent = '0';
    tagbar.className = 'unlabeled';
    tagify.removeAllTags();
    renderLabelsDropdown();
    renderGallery();
    return;
  }

  shot.src = `/image?path=${encodeURIComponent(item.input_path)}`;
  filepath.textContent = item.input_path;
  jump.textContent = filterMode ? '' : String(idx + 1);
  localStorage.setItem('tagger-index', String(idx));

  const tagged = (item.categories || []).length > 0;
  tagbar.className = tagged ? 'labeled' : 'unlabeled';

  tagify.off('add', onTagChange);
  tagify.off('remove', onTagChange);
  tagify.removeAllTags();
  if (tagged) tagify.addTags(item.categories);
  tagify.on('add', onTagChange);
  tagify.on('remove', onTagChange);

  tagify.settings.whitelist = [...new Set([...allTags, ...(item.categories || [])])];
  renderLabelsDropdown();
  renderGallery();
}

async function onTagChange() {
  if (saving) return;
  if (selectedPaths.size > 0) {
    await applyTagsToSelected().catch(error => {
      console.error(error);
    });
    return;
  }
  if (filterMode) {
    renderGallery();
    return;
  }
  const item = items[idx];
  if (!item) return;
  saving = true;
  try {
    const tags = tagify.value.map(entry => entry.value);
    item.categories = tags;
    const rowIdx = item._idx;
    if (Number.isInteger(rowIdx) && allItems[rowIdx]) {
      allItems[rowIdx].categories = tags;
    }
    tags.forEach(tag => {
      if (!allTags.includes(tag)) allTags.push(tag);
    });
    tagify.settings.whitelist = [...allTags];
    tagbar.className = tags.length ? 'labeled' : 'unlabeled';
    progress.textContent = `${labeledCount()} labeled`;
    const response = await fetch(`/api/item/${rowIdx}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ categories: tags }),
    });
    if (!response.ok) {
      throw new Error('single tag apply failed');
    }
  } finally {
    saving = false;
  }
}

function go(delta) {
  idx = Math.max(0, Math.min(items.length - 1, idx + delta));
  render();
}

function renderLabelsDropdown() {
  const counts = {};
  allItems.forEach(item => {
    (item.categories || []).forEach(category => {
      counts[category] = (counts[category] || 0) + 1;
    });
  });
  const labels = Object.keys(counts).sort((left, right) => {
    const diff = (counts[right] || 0) - (counts[left] || 0);
    if (diff !== 0) return diff;
    return left.localeCompare(right);
  });
  labelsDropdown.innerHTML = ['<option value="">labels</option>']
    .concat(labels.map(label => `<option value="${label}">${label} (${counts[label]})</option>`))
    .join('');
}

function renderGallery() {
  const inputElement = document.getElementById('tag-input');
  inputElement.placeholder = filterMode ? 'filter tags…' : 'add tags…';
  const clusterGallery = Boolean(selectedCluster);
  const selectionModeEnabled = filterMode || clusterGallery;
  const galleryEnabled = selectionModeEnabled && !forceSingleView;
  if (!galleryEnabled) {
    gallery.classList.add('hidden');
    gallery.innerHTML = '';
    shot.classList.remove('hidden');
    galleryIndex = -1;
    visibleGalleryItems = [];
    updateBulkBar(selectionModeEnabled, false);
    return;
  }
  const selected = tagify.value.map(entry => entry.value);
  let matches = [];
  if (selected.length) {
    matches = items.filter(item => selected.every(tag => (item.categories || []).includes(tag)));
  } else if (clusterGallery) {
    matches = items;
  } else {
    gallery.classList.add('hidden');
    gallery.innerHTML = '';
    shot.classList.remove('hidden');
    galleryIndex = -1;
    visibleGalleryItems = [];
    updateBulkBar(selectionModeEnabled, false);
    return;
  }
  visibleGalleryItems = matches;
  const visiblePaths = new Set(matches.map(item => item.input_path));
  selectedPaths = new Set([...selectedPaths].filter(path => visiblePaths.has(path)));
  shot.classList.add('hidden');
  gallery.classList.remove('hidden');
  gallery.classList.toggle('expanded', galleryExpanded);
  gallery.innerHTML = matches
    .map(item => (
      `<div class="thumb${selectedPaths.has(item.input_path) ? ' selected' : ''}" data-path="${item.input_path}">
        <button class="check" type="button">${selectedPaths.has(item.input_path) ? '✓' : ''}</button>
        <img src="/image?path=${encodeURIComponent(item.input_path)}" alt="">
      </div>`
    ))
    .join('');

  const thumbs = Array.from(gallery.querySelectorAll('.thumb'));
  const setThumbSelected = (thumb, isSelected) => {
    const check = thumb.querySelector('.check');
    thumb.classList.toggle('selected', isSelected);
    if (check) check.textContent = isSelected ? '✓' : '';
  };
  thumbs.forEach((thumb, thumbIdx) => {
    const image = thumb.querySelector('img');
    image.addEventListener('load', () => {
      thumb.style.aspectRatio = '';
    });

    const onHover = () => {
      galleryIndex = thumbIdx;
      thumbs.forEach(node => node.classList.remove('active'));
      thumb.classList.add('active');
      const path = matches[thumbIdx].input_path;
      filepath.textContent = path;
      jump.textContent = String(items.findIndex(item => item.input_path === path) + 1);
    };
    thumb.addEventListener('mouseenter', onHover);
    thumb.addEventListener('focus', onHover);
    thumb.addEventListener('click', event => {
      const path = matches[thumbIdx].input_path;
      const dotClicked = event.target.closest('.check') !== null;
      const additive = event.ctrlKey || event.metaKey || dotClicked;
      const ranged = event.shiftKey;
      if (ranged && selectionAnchorIndex >= 0) {
        const start = Math.min(selectionAnchorIndex, thumbIdx);
        const end = Math.max(selectionAnchorIndex, thumbIdx);
        for (let i = start; i <= end; i += 1) {
          const rangedPath = matches[i].input_path;
          selectedPaths.add(rangedPath);
          setThumbSelected(thumbs[i], true);
        }
      } else if (additive || !selectedCluster) {
        if (selectedPaths.has(path)) {
          selectedPaths.delete(path);
          setThumbSelected(thumb, false);
        } else {
          if (!additive && !ranged) {
            selectedPaths.clear();
            thumbs.forEach(node => setThumbSelected(node, false));
          }
          selectedPaths.add(path);
          setThumbSelected(thumb, true);
        }
        selectionAnchorIndex = thumbIdx;
      } else {
        const nextIndex = items.findIndex(item => item.input_path === path);
        if (nextIndex >= 0) {
          idx = nextIndex;
          forceSingleView = true;
          render();
          return;
        }
      }
      updateBulkBar(selectionModeEnabled, true);
    });
    thumb.addEventListener('dblclick', () => {
      const path = matches[thumbIdx].input_path;
      const nextIndex = items.findIndex(item => item.input_path === path);
      if (nextIndex < 0) return;
      idx = nextIndex;
      clearSelection();
      filterMode = false;
      filterToggle.classList.remove('active');
      render();
    });
  });

  if (thumbs.length) {
    galleryIndex = Math.min(Math.max(galleryIndex, 0), thumbs.length - 1);
    thumbs[galleryIndex].classList.add('active');
  }
  updateBulkBar(selectionModeEnabled, true);
}

document.addEventListener('keydown', event => {
  const inInput = tagify && document.activeElement === tagify.DOM.input;
  const inJump = document.activeElement === jump;
  if (inJump) return;

  if (inInput) {
    if (event.key === 'Escape') {
      tagify.DOM.input.blur();
      event.preventDefault();
    }
    if (event.key === 'Enter' && tagify.state.inputText === '') {
      tagify.DOM.input.blur();
      go(1);
      event.preventDefault();
    }
    return;
  }

  if (!gallery.classList.contains('hidden')) {
    if (event.key === 'j' || event.key === 'k') {
      const thumbs = Array.from(gallery.querySelectorAll('.thumb'));
      if (thumbs.length) {
        if (event.key === 'j') galleryIndex = Math.min(thumbs.length - 1, galleryIndex + 1);
        if (event.key === 'k') galleryIndex = Math.max(0, galleryIndex - 1);
        thumbs.forEach(node => node.classList.remove('active'));
        const target = thumbs[galleryIndex];
        target.classList.add('active');
        const path = target.getAttribute('data-path');
        filepath.textContent = path;
        jump.textContent = String(items.findIndex(item => item.input_path === path) + 1);
        event.preventDefault();
        return;
      }
    }
  }

  if (event.key === 'Escape' && forceSingleView) {
    forceSingleView = false;
    render();
    return;
  }

  if (event.key === 'j') go(1);
  if (event.key === 'k') go(-1);
  if (event.key === 'Enter' || event.key === '/') {
    tagify.DOM.input.focus();
    event.preventDefault();
  }
});

const nativeInput = document.getElementById('tag-input');
nativeInput.style.color = '#e8e8e8';
nativeInput.style.setProperty('--placeholder-color', '#909090');

init();
