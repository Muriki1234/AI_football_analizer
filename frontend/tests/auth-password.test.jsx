import {beforeEach,afterEach,describe,it,expect,vi} from 'vitest';
import {render,screen,fireEvent,waitFor,cleanup} from '@testing-library/react';
import {MemoryRouter,Routes,Route} from 'react-router-dom';
const mocks=vi.hoisted(()=>({updateUser:vi.fn(),finishRecovery:vi.fn(),signOut:vi.fn()}));
const state=vi.hoisted(()=>({user:{email:'verified@example.com'},loading:false,recovery:false}));
vi.mock('../src/lib/supabase',()=>({supabase:{auth:{updateUser:mocks.updateUser}}}));
vi.mock('../src/auth/AuthContext',()=>({useAuth:()=>({...state,...mocks})}));
import AccountMenu from '../src/components/AccountMenu';
import ResetPassword from '../src/pages/ResetPassword';
const show=()=>render(<MemoryRouter initialEntries={['/reset-password']}><Routes><Route path="/reset-password" element={<ResetPassword/>}/><Route path="/" element={<p>Home after password update</p>}/></Routes></MemoryRouter>);
const fill=(password,confirmation)=>{
 fireEvent.change(screen.getByLabelText('New password'),{target:{value:password}});
 fireEvent.change(screen.getByLabelText('Confirm password'),{target:{value:confirmation}});
 fireEvent.submit(screen.getByRole('button',{name:'Save new password'}).closest('form'));
};
describe('verified email password setup',()=>{
 beforeEach(()=>{vi.clearAllMocks();state.user={email:'verified@example.com'};mocks.updateUser.mockResolvedValue({error:null});});
 afterEach(cleanup);
 it('lets email-code customers reach password setup from their account',()=>{
  render(<MemoryRouter><AccountMenu/></MemoryRouter>);
  expect(screen.getByRole('link',{name:'Password'})).toHaveAttribute('href','/reset-password');
 });
 it('does not offer email-password sign-in settings to phone-only identities',()=>{
  state.user={phone:'64211234567'};render(<MemoryRouter><AccountMenu/></MemoryRouter>);
  expect(screen.queryByRole('link',{name:'Password'})).toBeNull();
 });
 it('requires matching passwords before calling the auth service',()=>{
  show();fill('ValidPassword!43','DifferentPassword!43');
  expect(screen.getByRole('alert')).toHaveTextContent('Passwords do not match');
  expect(mocks.updateUser).not.toHaveBeenCalled();
 });
 it('saves a password through Supabase and clears recovery only on success',async()=>{
  show();fill('ValidPassword!43','ValidPassword!43');
  expect(await screen.findByText('Home after password update')).toBeVisible();
  expect(mocks.updateUser).toHaveBeenCalledWith({password:'ValidPassword!43'});
  expect(mocks.finishRecovery).toHaveBeenCalledOnce();
 });
 it('keeps the form available when the auth service rejects the password',async()=>{
  mocks.updateUser.mockResolvedValue({error:{code:'weak_password'}});show();fill('RejectedPassword!43','RejectedPassword!43');
  await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('stronger password'));
  expect(mocks.finishRecovery).not.toHaveBeenCalled();
  expect(screen.getByRole('button',{name:'Save new password'})).toBeEnabled();
 });
 it('never shows a credential form without a verified login',()=>{
  state.user=null;show();expect(screen.getByRole('alert')).toHaveTextContent('invalid or expired');
  expect(screen.queryByLabelText('New password')).toBeNull();expect(mocks.updateUser).not.toHaveBeenCalled();
 });
});
