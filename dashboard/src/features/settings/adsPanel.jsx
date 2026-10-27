// ClippyMe — Settings → Ads: a sponsor banner shown for a time window and a
// mid-roll ad video, applied to every composed clip (download, publish, live
// monitor). Backed by /api/config/ads; the server clamps every value.
import { useState, useEffect } from 'react';
import { Icon, Btn, Switch, Segmented, Panel } from '../../components/primitives';
import { getAds, saveAds, uploadAdBanner, uploadAdVideo, deleteAdAsset } from '../../api/client';
import { setAdsActive } from '../../lib/adsState';

const POSITIONS = [
  { id: 'top-center', label: 'Top' }, { id: 'center', label: 'Center' },
  { id: 'bottom-center', label: 'Bottom' }, { id: 'top-right', label: 'Corner' },
];
const SIZES = [{ id: 'S', label: 'S' }, { id: 'M', label: 'M' }, { id: 'L', label: 'L' }];
const ANCHORS = [{ id: 'start', label: 'From start' }, { id: 'end', label: 'Before end' }];
const PLACEMENTS = [
  { id: 'smart', label: 'Mid-clip pause' }, { id: 'start', label: 'Before' }, { id: 'end', label: 'After' },
];

function NumField({ label, value, onCommit, min, max, step = 0.5 }) {
  const [v, setV] = useState(String(value ?? ''));
  useEffect(() => { setV(String(value ?? '')); }, [value]);
  return (
    <label className="od" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      {label}
      <input className="key-input" type="number" style={{ width: 72 }} min={min} max={max} step={step}
        aria-label={label} value={v} onChange={(e) => setV(e.target.value)}
        onBlur={() => { const n = Number(v); if (Number.isFinite(n) && n !== value) onCommit(n); }} />
      s
    </label>
  );
}

export function AdsPanel({ pushToast }) {
  const [ads, setAds] = useState(null);

  const apply = (status) => { setAds(status); setAdsActive(status); };
  useEffect(() => { getAds().then(apply).catch(() => {}); }, []);

  const save = async (patch) => {
    const next = {
      enabled: ads.enabled, banner: { ...ads.banner }, video: { ...ads.video }, ...patch,
    };
    try { apply(await saveAds(next)); }
    catch { pushToast?.('error', 'Saving ad settings failed'); }
  };
  const saveBanner = (p) => save({ banner: { ...ads.banner, ...p } });
  const saveVideo = (p) => save({ video: { ...ads.video, ...p } });

  const upload = (kind) => async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      await (kind === 'banner' ? uploadAdBanner(f) : uploadAdVideo(f));
      apply(await getAds());
      pushToast?.('success', kind === 'banner' ? 'Ad banner uploaded' : 'Ad video uploaded');
    } catch (err) { pushToast?.('error', String(err.message || 'Upload failed').slice(0, 80)); }
  };
  const remove = (kind) => async () => {
    try { await deleteAdAsset(kind); apply(await getAds()); }
    catch { pushToast?.('error', 'Remove failed'); }
  };

  if (!ads) return null;
  const b = ads.banner;
  const v = ads.video;
  return (
    <Panel title="Ads" sub="Sponsor banner and ad insert, added to every clip automatically" icon="layers"
      style={{ marginBottom: 18 }}
      headRight={<Switch on={ads.enabled} label="Enable ads" onChange={(on) => save({ enabled: on })} />}>
      <div className="opt" style={{ alignItems: 'flex-start' }}>
        <div className="oico"><Icon n="stamp" /></div>
        <div className="otxt" style={{ flex: 1 }}>
          <div className="ot">Timed banner</div>
          <div className="od">
            {ads.banner_uploaded ? 'PNG shown for a set time. For a banner on the whole clip use the Logo layer.'
              : 'Upload a transparent PNG'}
          </div>
          {ads.banner_uploaded && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10, alignItems: 'center' }}>
              <Segmented label="Count from" value={b.anchor} options={ANCHORS} onChange={(id) => saveBanner({ anchor: id })} />
              <NumField label="Offset" value={b.start} min={0} max={600} onCommit={(n) => saveBanner({ start: n })} />
              <NumField label="Show for" value={b.duration} min={0.5} max={600} onCommit={(n) => saveBanner({ duration: n })} />
              <Segmented label="Position" value={b.position} options={POSITIONS} onChange={(id) => saveBanner({ position: id })} />
              <Segmented label="Size" value={b.size} options={SIZES} onChange={(id) => saveBanner({ size: id })} />
            </div>
          )}
        </div>
        <div className="r" style={{ gap: 8 }}>
          <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
            <Icon n="upload" />Upload
            <input type="file" accept="image/png,.png" hidden onChange={upload('banner')} />
          </label>
          {ads.banner_uploaded && <Btn variant="ghost" size="sm" icon="trash-2" onClick={remove('banner')}>Remove</Btn>}
          <Switch on={b.enabled} label="Enable timed banner" disabled={!ads.banner_uploaded}
            onChange={(on) => saveBanner({ enabled: on })} />
        </div>
      </div>
      <div className="opt" style={{ borderBottom: 0, alignItems: 'flex-start' }}>
        <div className="oico"><Icon n="file-video" /></div>
        <div className="otxt" style={{ flex: 1 }}>
          <div className="ot">Ad insert</div>
          <div className="od">
            {ads.video_uploaded
              ? `The clip pauses, the ad plays with its own sound (${ads.video_duration ?? '?'} s), then the clip continues. Mid-clip picks a pause between phrases, ideally right after a laugh.`
              : 'Upload a video (up to 3 min); any shape is fitted to the clip frame'}
          </div>
          {ads.video_uploaded && (
            <div style={{ marginTop: 10 }}>
              <Segmented label="Ad placement" value={v.placement} options={PLACEMENTS} onChange={(id) => saveVideo({ placement: id })} />
            </div>
          )}
        </div>
        <div className="r" style={{ gap: 8 }}>
          <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
            <Icon n="upload" />Upload
            <input type="file" accept="video/*" hidden onChange={upload('video')} />
          </label>
          {ads.video_uploaded && <Btn variant="ghost" size="sm" icon="trash-2" onClick={remove('video')}>Remove</Btn>}
          <Switch on={v.enabled} label="Enable ad insert" disabled={!ads.video_uploaded}
            onChange={(on) => saveVideo({ enabled: on })} />
        </div>
      </div>
    </Panel>
  );
}
