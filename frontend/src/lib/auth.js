export const isCustomer = (user) => Boolean(
    user && !user.is_anonymous && (user.email_confirmed_at || user.phone_confirmed_at)
);

export const safeReturnPath = (value) => {
    if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || /[\\\s]/.test(value)) return '/';
    try {
        const url = new URL(value, 'https://app.invalid');
        if (url.origin !== 'https://app.invalid' || /^\/(login|auth|reset-password)(\/|$)/.test(url.pathname)) return '/';
        return url.pathname + url.search;
    } catch { return '/'; }
};

export function normalizePhone(number, prefix = '+86') {
    const input = number.trim().replace(/[\s()-]/g, '');
    const phone = input.startsWith('+') ? input : prefix + input.replace(/^0+/, '');
    if (!/^\+[1-9]\d{7,14}$/.test(phone)) throw new Error('Enter a valid mobile number with its country code.');
    if (phone.startsWith('+86') && !/^\+861[3-9]\d{9}$/.test(phone)) throw new Error('Enter a valid mainland China mobile number.');
    return phone;
}

export function authErrorMessage(error) {
    const code = error?.code;
    if (code === 'invalid_credentials') return 'Email or password is incorrect.';
    if (code === 'email_not_confirmed') return 'Verify your email first, or sign in with an email code.';
    if (['otp_expired', 'otp_disabled'].includes(code)) return 'The code is invalid or expired. Request a new code.';
    if (code === 'over_email_send_rate_limit' || code === 'over_sms_send_rate_limit' || code === 'over_request_rate_limit' || error?.status === 429) return 'Too many requests. Please wait before trying again.';
    if (['email_provider_disabled', 'phone_provider_disabled', 'sms_send_failed', 'email_address_not_authorized'].includes(code)) return 'This sign-in method is temporarily unavailable. Please contact support.';
    if (code === 'weak_password') return 'Use a stronger password with at least 10 characters.';
    if (error?.name === 'AuthRetryableFetchError' || error instanceof TypeError) return 'Unable to connect. Check your connection and try again.';
    return 'Unable to sign in. Please try again or contact support.';
}
