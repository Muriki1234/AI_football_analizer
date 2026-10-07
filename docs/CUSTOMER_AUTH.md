# 客户登录交付与开通

邮箱验证码、国内/海外手机号验证码、已有邮箱账号密码登录、密码恢复、路由保护、跨标签页状态同步、按用户缓存与退出已实现。验证码由 Supabase Auth 生成、验证和过期，密码由 Supabase 保存，应用不保存明文密码或验证码。

**目前不能宣称已可交付客户：生产 SMTP 和短信服务尚未开通，真实收码尚未验证。** `AUTH_EMAIL_OTP_ENABLED`、`AUTH_PHONE_OTP_ENABLED` 默认关闭；界面提示未开放，不假装发送成功。现有密码账号仍可登录。

## 邮箱：优先开通免费额度

1. 注册 [Resend](https://resend.com/pricing) 的免费账户，目前为 3,000 封/月、100 封/日。注册条款、账号凭据由项目所有者本人处理，不开启自动付费超额。
2. 验证 `aifootnova.com` 的发信域名，按 Resend 给出的记录配置 DNS。只添加所需发信记录，不替换现有网页域名记录；保留已有 MX。
3. 在 [Supabase SMTP](https://supabase.com/dashboard/project/dexhzatnsyebupowbbmt/auth/smtp) 配置 Resend SMTP。按 [Resend 官方说明](https://resend.com/docs/send-with-supabase-smtp)：主机 `smtp.resend.com`，端口 `465`，用户名 `resend`，密码为 Resend API key，发件人使用已验证域名，如 `no-reply@aifootnova.com`。密钥只存提供商或 Vercel Secret，不放到聊天、代码或 VITE 变量。
4. 将 `docs/auth/email-code.html` 用于 Supabase 的 **Magic link or OTP** 与 **Confirm sign up** 邮件模板；两种都包含 `{{ .Token }}`，新老用户都能看到验证码。密码恢复模板使用原生 `{{ .ConfirmationURL }}`。
5. [Auth URL Configuration](https://supabase.com/dashboard/project/dexhzatnsyebupowbbmt/auth/url-configuration)：Site URL 为 `https://www.aifootnova.com`，仅允许明确的 `/auth/callback`、`/reset-password` 和首页；预览需要测试邮件链接时添加精确预览 URL，不允许任意站点通配符。
6. 邮箱 OTP 过期建议 600 秒；保留邮箱确认；发送频率、IP 限额使用 Supabase 服务端设置。验证 Gmail 与至少一个国内常用邮箱收到验证码、错误码被拒绝、验证码不能重用、退出和刷新正常后，将 Vercel server env `AUTH_EMAIL_OTP_ENABLED=true` 并重新部署。

## 短信：一个腾讯云应用，分别配置国内/海外模板

代码使用腾讯云官方短信 SDK，国内 `+86` 使用国内模板，海外及港澳台使用国际模板。当前[国内短信要求](https://cloud.tencent.com/document/product/382/13444)包括企业资质/企业授权和运营商签名实名报备，个人网站无法直接保证获批。模板需审核并生效。[发送 API](https://cloud.tencent.com/document/api/382/55981)对国内及国际使用不同模板；国际支持范围与送达需按目标国家测试。

1. 所有者开通服务并完成必要资质审核、充值/付费与预算授权。验证码模板只设置一个变量，例如「您的 FootNova 登录验证码为 {1}，请勿泄露」。国际模板例如 `Your FootNova verification code is {1}. Do not share it.`
2. 用仅具备短信发送权限的 CAM 子账号密钥，在 Vercel server env 配置：`TENCENT_SMS_SECRET_ID`、`TENCENT_SMS_SECRET_KEY`、`TENCENT_SMS_APP_ID`、`TENCENT_SMS_REGION`、`TENCENT_SMS_TEMPLATE_CN`、`TENCENT_SMS_SIGN_CN`、`TENCENT_SMS_TEMPLATE_INTL`，以及可选国际签名/ SenderId。
3. 在 [Supabase Auth Hooks](https://supabase.com/dashboard/project/dexhzatnsyebupowbbmt/auth/hooks) 配置 **Send SMS HTTP Hook**，URL 为 `https://www.aifootnova.com/api/auth-sms-hook`。将该 Hook 的签名 secret 配置为 Vercel `AUTH_SMS_HOOK_SECRET`。代码保留原始请求体并用 standardwebhooks 验签，拒绝过期/伪造请求，不打印手机号、OTP 或密钥。
4. 开启 Phone provider，保留 phone confirmation，OTP 长度设为 6、过期建议 300 秒。不要配置固定测试验证码到生产。
5. `AUTH_SMS_ALLOWED_PREFIXES` 设置为实际客户需要的国家区号（如 `+86,+64,+61`），无需发送的目的地不开放。设置 Supabase 发送限额、腾讯云每号码发送频率限制和账号每日费用/发送量限额。若公开注册出现滥用，开通 Supabase 支持的 CAPTCHA 后再扩大发送量；界面倒计时不是安全限流。
6. 在受控环境验证国内以及每个准备开放的海外区域收码，错误和过期验证码、重发、退出均正确后再在生产设置 `AUTH_PHONE_OTP_ENABLED=true`。不要仅凭 SDK 返回 `Ok` 宣称真实收件成功。

邮箱与手机号当前是不同登录身份，可能生成不同账户。页面明确提示客户持续使用同一方式；不要根据客户端输入或昵称自动合并用户数据。账号绑定可在验证两种身份的专门流程中后续实现。

## 后端权限

迁移 `20261007064550_customer_auth_isolation.sql`：检查有效 `auth.sessions`、已确认邮箱/手机号、拒绝匿名身份，替换旧 permissive RLS，保护 sessions/tasks/旧 Supabase 视频；撤销 PUBLIC、anon、authenticated 对特权更新/清理 RPC 的权限，保留 service_role worker 权限。没有删除用户或分析记录。不要通过关闭 RLS 回滚。

Vercel 校验 Supabase JWT、有效登录会话以及明确的 `session.user_id`。RunPod 状态结果需要服务端签发的 `status_token`，绑定用户、session、完整任务 ID、有效期；前端通过 `X-Job-Token` 传递，不允许任意登录用户查询其他客户任务。上传对象路径包含用户 ID、session ID 和额外随机 ID，避免覆盖他人上传。RunPod 密钥与短信凭据仅在服务器。

**媒体访问边界：现有 R2 bucket 使用公开媒体地址。数据库隔离不会撤销已分享的媒体 URL；知道完整 URL 的人仍可能读取文件。** 本次登录改动不改变 R2 bucket 的公开设置，也不宣称视频具有登录后即时可撤销的访问控制。若客户要求私密文件，需要迁移到私有 bucket、为用户和 worker 生成短时签名 URL，并验证全部视频/图表读取路径。

## 验证与发布

本地执行：`npm run test -- tests/auth-utils.test.js tests/auth-api.test.js tests/auth-ui.test.jsx tests/auth-sms-hook.test.js`，再执行构建与修改文件 ESLint。使用干净分支发布，排除工作区原有未提交的 pipeline/UI 实验。

线上测试使用隔离的临时测试身份，实际 Supabase Auth 登录 → Vercel API → Supabase RLS；确认两个账号互不可读写，匿名/未登录请求返回 401/403，退出后旧 JWT 被拒绝，service_role worker 仍能读取自己的任务。认证测试不必启动收费 GPU。若验证分析链路，使用短视频，另外检查 RunPod job/worker logs，不能将认证验证等同于整个视频分析验证。
