import assert from 'node:assert/strict';
export async function waitThenJudge(name,read,judge,{timeoutMs=20000,intervalMs=400,clock=Date.now,sleep=ms=>new Promise(r=>setTimeout(r,ms)),readWithin=async(read,remaining)=>{let timer;try{return await Promise.race([read(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('PAGE_READ_DEADLINE')),remaining);})]);}finally{clearTimeout(timer);}},record=()=>{}}={}){
 const start=clock(),polls=[];let last,lastCompletedFailure=null,deadlineCause=null;
 const finish=(status)=>({name,status,timeoutMs,startedAtUtc:new Date(start).toISOString(),endedAtUtc:new Date(clock()).toISOString(),elapsedMs:clock()-start,polls,last,lastCompletedFailure,deadlineCause});
 while(true){
  const remaining=timeoutMs-(clock()-start);
  if(remaining<=0){deadlineCause='WAIT_DEADLINE';const proof=finish('FAILED_BOUNDED_READINESS');record(proof);throw Error('READINESS_TIMEOUT '+name+' '+(lastCompletedFailure||deadlineCause)+'; deadline='+deadlineCause);}
  const at=clock();let completed=false;
  try{
   last=await readWithin(read,remaining);completed=true;
   if(last?.blocker)throw Error('OWNER_ONLY_PROMPT');
   judge(last);
   if(clock()-start>timeoutMs)throw Error('WAIT_DEADLINE');
   polls.push({atUtc:new Date(at).toISOString(),ready:true,completed:true});
   const proof={...finish('PASS'),state:last};record(proof);return proof;
  }catch(e){
   polls.push({atUtc:new Date(at).toISOString(),ready:false,completed,result:e.message});
   if(completed&&!['WAIT_DEADLINE','PAGE_READ_DEADLINE'].includes(e.message))lastCompletedFailure=e.message;
   if(/OWNER_ONLY_PROMPT/.test(e.message)){const proof=finish('STOP_OWNER_ONLY_PROMPT');record(proof);throw e;}
   if(['PAGE_READ_DEADLINE','WAIT_DEADLINE'].includes(e.message)||clock()-start>=timeoutMs){deadlineCause=e.message;const proof=finish('FAILED_BOUNDED_READINESS');record(proof);throw Error('READINESS_TIMEOUT '+name+' '+(lastCompletedFailure||e.message)+'; deadline='+deadlineCause);}
   await sleep(Math.min(intervalMs,timeoutMs-(clock()-start)));
  }
 }
}
