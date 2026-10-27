// Whether the global ads layer will render on composed clips. Ads are a
// server-side setting (Settings → Ads), not a per-clip toggle, so a clip with
// every layer off must still go through compose to get them. Settings and the
// app shell keep this in sync with GET /api/config/ads.
let active = false;

export function adsWillRender(status) {
  if (!status?.enabled) return false;
  return !!((status.banner?.enabled && status.banner_uploaded)
    || (status.video?.enabled && status.video_uploaded));
}

export function setAdsActive(status) { active = adsWillRender(status); }

export function adsActive() { return active; }
