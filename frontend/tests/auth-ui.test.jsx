import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
const auth = vi.hoisted(() => ({ signInWithOtp: vi.fn(), verifyOtp: vi.fn(), signInWithPassword: vi.fn(), resetPasswordForEmail: vi.fn(), signUp: vi.fn(), resend: vi.fn() }));
const state = vi.hoisted(() => ({ user: null, loading: false, recovery: false }));
vi.mock('../src/lib/supabase', () => ({ supabase: { auth } }));
vi.mock('../src/auth/AuthContext', () => ({ useAuth: () => state }));
vi.mock('react-icons/hi2', () => ({ HiEnvelope: () => null, HiLockClosed: () => null, HiEye: () => null, HiEyeSlash: () => null, HiDevicePhoneMobile: () => null }));
import Login from '../src/pages/Login';
import RequireAuth from '../src/auth/RequireAuth';

const showLogin = () => render(<MemoryRouter initialEntries={['/login']}><Login /></MemoryRouter>);
const fillEmail = () => fireEvent.change(screen.getByLabelText('Email'), { target: { value: ' Person@Example.com ' } });

describe('customer email sign-in', () => {
    beforeEach(() => {
        vi.clearAllMocks(); Object.assign(state, { user: null, loading: false, recovery: false });
        Object.values(auth).forEach((mock) => mock.mockResolvedValue({ error: null }));
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ emailOtp: true, phoneOtp: true, password: true }) }));
    });
    afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
    it('sends an email code and verifies the same normalized email', async () => {
        showLogin(); fireEvent.click(screen.getByRole('button', { name: 'Email code' })); await waitFor(() => expect(screen.getByRole('button', { name: 'Send verification code' })).toBeEnabled());
        fillEmail(); fireEvent.submit(screen.getByRole('button', { name: 'Send verification code' }).closest('form'));
        await screen.findByLabelText('Verification code');
        expect(auth.signInWithOtp).toHaveBeenCalledWith({ email: 'person@example.com', options: { shouldCreateUser: true, emailRedirectTo: expect.stringContaining('/auth/callback') } });
        fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        await waitFor(() => expect(auth.verifyOtp).toHaveBeenCalledWith({ email: 'person@example.com', token: '123456', type: 'email' }));
        expect(screen.getByRole('button', { name: /Resend code in/ })).toBeDisabled();
    });
    it('defaults to password sign-in and does not offer SMS', async () => {
        showLogin();
        expect(screen.getByRole('button', { name: 'Password', exact: true })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.queryByRole('button', { name: 'Phone code' })).toBeNull();
        fillEmail(); fireEvent.change(screen.getByLabelText('Password', { exact: true }), { target: { value: 'StrongPassword123!' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Sign in', exact: true }).closest('form'));
        await waitFor(() => expect(auth.signInWithPassword).toHaveBeenCalledWith({ email: 'person@example.com', password: 'StrongPassword123!' }));
        expect(auth.signInWithOtp).not.toHaveBeenCalled();
    });
    it('registers with a password and requires email confirmation', async () => {
        showLogin(); fireEvent.click(screen.getByRole('button', { name: /New to FootNova/ }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Create account and send code' })).toBeEnabled());
        fillEmail();
        fireEvent.change(screen.getByLabelText('Password', { exact: true }), { target: { value: 'StrongPassword123!' } });
        fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: 'StrongPassword123!' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Create account and send code' }).closest('form'));
        await screen.findByLabelText('Verification code');
        expect(auth.signUp).toHaveBeenCalledWith({ email: 'person@example.com', password: 'StrongPassword123!', options: { emailRedirectTo: expect.stringContaining('/auth/callback') } });
        expect(auth.signInWithPassword).not.toHaveBeenCalled();
        expect(screen.queryByLabelText('Password', { exact: true })).toBeNull();
        expect(screen.queryByLabelText('Email')).toBeNull();
        fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        await waitFor(() => expect(auth.verifyOtp).toHaveBeenCalledWith({ email: 'person@example.com', token: '123456', type: 'email' }));
        expect(screen.getByRole('button', { name: /Resend code in/ })).toBeDisabled();
    });
    it('resends signup confirmation without repeating signup or retaining the password', async () => {
        const clock = vi.spyOn(Date, 'now').mockReturnValue(1000);
        try {
            showLogin(); fireEvent.click(screen.getByRole('button', { name: /New to FootNova/ }));
            await waitFor(() => expect(screen.getByRole('button', { name: 'Create account and send code' })).toBeEnabled());
            fillEmail();
            fireEvent.change(screen.getByLabelText('Password', { exact: true }), { target: { value: 'StrongPassword123!' } });
            fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: 'StrongPassword123!' } });
            fireEvent.submit(screen.getByRole('button', { name: 'Create account and send code' }).closest('form'));
            await screen.findByLabelText('Verification code');
            clock.mockReturnValue(62000);
            await waitFor(() => expect(screen.getByRole('button', { name: 'Resend code', exact: true })).toBeEnabled(), { timeout: 2200 });
            fireEvent.click(screen.getByRole('button', { name: 'Resend code', exact: true }));
            await waitFor(() => expect(auth.resend).toHaveBeenCalledWith({ type: 'signup', email: 'person@example.com', options: { emailRedirectTo: expect.stringContaining('/auth/callback') } }));
            expect(auth.signUp).toHaveBeenCalledTimes(1);
            expect(auth.signInWithOtp).not.toHaveBeenCalled();
        } finally { clock.mockRestore(); }
    });
    it.each([
        ['short', 'short', 'Use at least 10 characters.'],
        ['StrongPassword123!', 'AnotherPassword123!', 'Passwords do not match.'],
    ])('rejects invalid signup passwords before calling Supabase: %s', async (password, confirmation, message) => {
        showLogin(); fireEvent.click(screen.getByRole('button', { name: /New to FootNova/ }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Create account and send code' })).toBeEnabled());
        fillEmail(); fireEvent.change(screen.getByLabelText('Password', { exact: true }), { target: { value: password } });
        fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: confirmation } });
        fireEvent.submit(screen.getByRole('button', { name: 'Create account and send code' }).closest('form'));
        expect(await screen.findByRole('alert')).toHaveTextContent(message);
        expect(auth.signUp).not.toHaveBeenCalled();
    });
    it('keeps the code form usable after an expired code', async () => {
        auth.verifyOtp.mockResolvedValue({ error: { code: 'otp_expired' } }); showLogin(); fireEvent.click(screen.getByRole('button', { name: 'Email code' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Send verification code' })).toBeEnabled());
        fillEmail(); fireEvent.submit(screen.getByRole('button', { name: 'Send verification code' }).closest('form'));
        fireEvent.change(await screen.findByLabelText('Verification code'), { target: { value: '123456' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Verify and sign in' }).closest('form'));
        expect(await screen.findByRole('alert')).toHaveTextContent('invalid or expired');
        expect(screen.getByRole('button', { name: 'Verify and sign in' })).toBeEnabled();
    });
    it('does not send when the production provider is unconfigured', async () => {
        fetch.mockResolvedValue({ ok: true, json: async () => ({ emailOtp: false, phoneOtp: false, password: true }) });
        showLogin(); fireEvent.click(screen.getByRole('button', { name: 'Email code' })); expect(await screen.findByText(/Email verification is not available yet/)).toBeVisible();
        expect(screen.getByRole('button', { name: 'Send verification code' })).toBeDisabled();
        expect(auth.signInWithOtp).not.toHaveBeenCalled();
    });
    it('fails closed when sign-in method configuration cannot load', async () => {
        fetch.mockResolvedValue({ ok: false }); showLogin(); fireEvent.click(screen.getByRole('button', { name: 'Email code' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load sign-in methods');
        expect(screen.getByRole('button', { name: 'Send verification code' })).toBeDisabled();
    });
    it('blocks signup without a production email provider but keeps password login available', async () => {
        fetch.mockResolvedValue({ ok: true, json: async () => ({ emailOtp: false, phoneOtp: false, password: true }) });
        showLogin(); expect(screen.getByRole('button', { name: 'Sign in', exact: true })).toBeEnabled();
        fireEvent.click(screen.getByRole('button', { name: /New to FootNova/ }));
        expect(await screen.findByText(/Email verification is not available yet/)).toBeVisible();
        const submit = screen.getByRole('button', { name: 'Create account and send code' });
        expect(submit).toBeDisabled();
        fireEvent.submit(submit.closest('form'));
        expect(auth.signUp).not.toHaveBeenCalled();
    });
    it('retains the signup form after provider failure and does not claim a code was sent', async () => {
        auth.signUp.mockResolvedValue({ error: { code: 'email_address_not_authorized' } });
        showLogin(); fireEvent.click(screen.getByRole('button', { name: /New to FootNova/ }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Create account and send code' })).toBeEnabled());
        fillEmail(); fireEvent.change(screen.getByLabelText('Password', { exact: true }), { target: { value: 'StrongPassword123!' } });
        fireEvent.change(screen.getByLabelText('Confirm password'), { target: { value: 'StrongPassword123!' } });
        fireEvent.submit(screen.getByRole('button', { name: 'Create account and send code' }).closest('form'));
        expect(await screen.findByRole('alert')).toHaveTextContent('temporarily unavailable');
        expect(screen.queryByLabelText('Verification code')).toBeNull();
        expect(screen.getByLabelText('Email')).toBeVisible();
    });
    it('requires authentication before rendering private analysis pages', async () => {
        render(<MemoryRouter initialEntries={['/dashboard?sessionId=private']}><Routes><Route path="/dashboard" element={<RequireAuth><p>Private analysis</p></RequireAuth>} /><Route path="/login" element={<p>Sign in required</p>} /></Routes></MemoryRouter>);
        expect(await screen.findByText('Sign in required')).toBeVisible(); expect(screen.queryByText('Private analysis')).toBeNull();
    });
});
