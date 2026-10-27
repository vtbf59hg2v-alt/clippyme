import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AdsPanel } from './adsPanel';
import { adsActive } from '../../lib/adsState';

const getAds = vi.fn();
const saveAds = vi.fn();
vi.mock('../../api/client', () => ({
  getAds: (...a) => getAds(...a),
  saveAds: (...a) => saveAds(...a),
  uploadAdBanner: vi.fn(),
  uploadAdVideo: vi.fn(),
  deleteAdAsset: vi.fn(),
}));

const STATUS = {
  enabled: false,
  banner: { enabled: true, anchor: 'start', start: 3, duration: 8, position: 'bottom-center', size: 'M' },
  video: { enabled: true, placement: 'smart' },
  banner_uploaded: true,
  video_uploaded: true,
  video_duration: 12.5,
};

beforeEach(() => { vi.clearAllMocks(); getAds.mockResolvedValue(STATUS); });

describe('AdsPanel', () => {
  it('shows the uploaded ad and its placement', async () => {
    render(<AdsPanel />);
    expect(await screen.findByText(/12.5 s/)).toBeTruthy();
    expect(screen.getByText('Mid-clip pause')).toBeTruthy();
  });

  it('master switch saves the whole config and updates the global flag', async () => {
    saveAds.mockResolvedValue({ ...STATUS, enabled: true });
    render(<AdsPanel />);
    fireEvent.click(await screen.findByRole('switch', { name: 'Enable ads' }));
    await waitFor(() => expect(saveAds).toHaveBeenCalled());
    expect(saveAds.mock.calls[0][0]).toMatchObject({ enabled: true, banner: { start: 3 }, video: { placement: 'smart' } });
    await waitFor(() => expect(adsActive()).toBe(true));
  });

  it('banner timing commits on blur', async () => {
    saveAds.mockResolvedValue(STATUS);
    render(<AdsPanel />);
    const field = await screen.findByLabelText('Show for');
    fireEvent.change(field, { target: { value: '5' } });
    fireEvent.blur(field);
    await waitFor(() => expect(saveAds).toHaveBeenCalled());
    expect(saveAds.mock.calls[0][0].banner.duration).toBe(5);
  });
});
