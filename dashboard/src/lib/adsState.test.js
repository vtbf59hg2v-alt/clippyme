import { describe, it, expect } from 'vitest';
import { adsWillRender, setAdsActive, adsActive } from './adsState';

describe('adsState', () => {
  it('needs the master switch, a part switch and its asset', () => {
    expect(adsWillRender(null)).toBe(false);
    expect(adsWillRender({ enabled: false, banner: { enabled: true }, banner_uploaded: true })).toBe(false);
    expect(adsWillRender({ enabled: true, banner: { enabled: true }, banner_uploaded: false })).toBe(false);
    expect(adsWillRender({ enabled: true, banner: { enabled: true }, banner_uploaded: true })).toBe(true);
    expect(adsWillRender({ enabled: true, video: { enabled: true }, video_uploaded: true })).toBe(true);
  });

  it('remembers the last status', () => {
    setAdsActive({ enabled: true, video: { enabled: true }, video_uploaded: true });
    expect(adsActive()).toBe(true);
    setAdsActive({ enabled: false });
    expect(adsActive()).toBe(false);
  });
});
