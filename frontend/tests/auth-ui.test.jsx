import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
const auth = vi.hoisted(() => ({ signInWithOtp: vi.fn(), verifyOtp: vi.fn(), signInWithPassword: vi.fn(), resetPasswordForEmail: vi.fn() }));
const state = vi.hoisted(() => ({ user: null, loading: false, recovery: false }));
vi.mock('../src/lib/supabase', () => ({ supabase: { auth } }));
vi.mock('../src/auth/AuthContext', () => ({ useAuth: () => state }));
vi.mock('react-icons/hi2', () => ({ HiEnvelope: () => null, HiLockClosed: () => null, HiEye: () => null, HiEyeSlash: () => null, HiDevicePhoneMobile: () => null }));
import Login from '../src/pages/Login';
import RequireAuth from '../src/auth/RequireAuth';

const showLogin = () => render(<MemoryRouter initialEntries={['/login']}><Login /></MemoryRouter>);
const fillEmail = () => fireEvent.change(screen.getByLabelText('Email'), { target: { value: ' Person@Example.com ' } });

describe('verification code sign-in', () => {
    beforeEach(() => {
        vi.clearAllMocks(); Object.assign(state, { user: null, loading: false, recovery: false });
        auth.signInWithOtp.mockResolvedValue({ error: null }); auth.verifyOtp.mockResolvedValue({ error: null });
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ emailOtp: true, phoneOtp: true, password: true }) }));
    });
    afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
    it('sends an email code and verifies the same normalized email', async () => {
        showLogin(); await waitFor(() => expect(screen.getByRole('button', { name: 'Send verification code' })).toBeEnabled());
        fillEmail(); fireEvent.submit(screen.getByRole('button', { name: 'Send verification code' }).closest('form'));
        await screen.findByLabelText('Verification code');
        expect(auth.signInWithOtp).toHaveBeenCalledWith({ email: 'person@example.com', options: { shouldCreateUser: true, emailRedirectTo: expect.stringContaining('/auth/callback') } });
        fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        await waitFor(() => expect(auth.verifyOtp).toHaveBeenCalledWith({ email: 'person@example.com', token: '123456', type: 'email' }));
        expect(screen.getByRole('button', { name: /Resend code in/ })).toBeDisabled();
    });
    it('sends international SMS codes with the selected country prefix', async () => {
        showLogin(); fireEvent.click(screen.getByRole('button', { name: 'Phone code' }));
        fireEvent.change(screen.getByLabelText('Country / region'), { target: { value: '+64' } });
        fireEvent.change(screen.getByLabelText('Mobile number'), { target: { value: '021 123 4567' } });
        await waitFor(() => expect(screen.getByRole('button', { name: 'Send verification code' })).toBeEnabled());
        fireEvent.submit(screen.getByRole('button', { name: 'Send verification code' }).closest('form'));
        await screen.findByLabelText('Verification code');
        expect(auth.signInWithOtp).toHaveBeenCalledWith({ phone: '+64211234567', options: { shouldCreateUser: true, channel: 'sms' } });
        fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        await waitFor(() => expect(auth.verifyOtp).toHaveBeenCalledWith({ phone: '+64211234567', token: '123456', type: 'sms' }));
    });
    it('keeps the code form usable after an expired code', async () => {
        auth.verifyOtp.mockResolvedValue({ error: { code: 'otp_expired' } }); showLogin();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Send verification code' })).toBeEnabled());
        fillEmail(); fireEvent.submit(screen.getByRole('button', { name: 'Send verification code' }).closest('form'));
        fireEvent.change(await screen.findByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        expect(await screen.findByRole('alert')).toHaveTextContent('invalid or expired');
        expect(screen.getByRole('button', { name: 'Verify and sign in' })).toBeEnabled();
    });
    it('does not send when the production provider is unconfigured', async () => {
        fetch.mockResolvedValue({ ok: true, json: async () => ({ emailOtp: false, phoneOtp: false, password: true }) });
        showLogin(); expect(await screen.findByText(/Email codes are not available yet/)).toBeVisible();
        expect(screen.getByRole('button', { name: 'Send verification code' })).toBeDisabled();
        expect(auth.signInWithOtp).not.toHaveBeenCalled();
    });
    it('fails closed when sign-in method configuration cannot load', async () => {
        fetch.mockResolvedValue({ ok: false }); showLogin();
        expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load sign-in methods');
        expect(screen.getByRole('button', { name: 'Send verification code' })).toBeDisabled();
    });
    it('requires authentication before rendering private analysis pages', async () => {
        render(<MemoryRouter initialEntries={['/dashboard?sessionId=private']}><Routes><Route path="/dashboard" element={<RequireAuth><p>Private analysis</p></RequireAuth>} /><Route path="/login" element={<p>Sign in required</p>} /></Routes></MemoryRouter>);
        expect(await screen.findByText('Sign in required')).toBeVisible(); expect(screen.queryByText('Private analysis')).toBeNull();
    });
});
