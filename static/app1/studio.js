const form = document.querySelector('#detect-form');
const feedback = document.querySelector('#feedback-form');
const fields = [...form.querySelectorAll('input:not([type=hidden])')];
function syncConfig() {
  const target = document.querySelector('#feedback-config'); target.replaceChildren();
  fields.forEach(input => {const copy = document.createElement('input'); copy.type='hidden';copy.name=input.name;copy.value=input.value;target.append(copy);});
}
syncConfig();
fields.forEach(input=>input.addEventListener('input',()=>{
  document.querySelectorAll('.current-result').forEach(el=>el.hidden=true);
  document.querySelector('#feedback-run').value='';
  document.querySelector('#new-feedback-run').value='';
  document.querySelector('#empty-result').hidden=false;
  document.querySelector('#changed').hidden=false; syncConfig();
}));
feedback.addEventListener('submit',syncConfig);
form.addEventListener('submit',async event=>{
  event.preventDefault();
  const payload = new FormData(form);
  const button = document.querySelector('#detect-button');
  const error = document.querySelector('#error');
  const progress = document.querySelector('#progress');
  const status = document.querySelector('#progress-text');
  button.disabled=true; fields.forEach(el=>el.disabled=true);
  feedback.querySelector('button').disabled=true;
  error.hidden=true; document.querySelector('#progress-area').hidden=false;
  document.querySelectorAll('.current-result').forEach(el=>el.hidden=true);
  document.querySelector('#feedback-run').value='';
  document.querySelector('#new-feedback-run').value='';
  document.querySelector('#empty-result').hidden=false;
  progress.value=0; status.textContent='Đang nạp model…';
  try {
    const response = await fetch(form.action,{method:'POST',body:payload});
    if (!response.ok) {let text='Không xử lý được ảnh.';try{text=(await response.json()).error||text;}catch{}throw new Error(text);}
    const reader=response.body.getReader(); const decoder=new TextDecoder();let buffer='';let complete=false;
    while(true){
      const {value,done}=await reader.read(); if(done)break;
      buffer+=decoder.decode(value,{stream:true}); const lines=buffer.split('\n');buffer=lines.pop();
      for(const line of lines){
        if(!line.trim())continue;const data=JSON.parse(line);
        if(data.error)throw new Error(data.error);
        if(data.status==='progress'){progress.value=data.total?data.done/data.total:0;status.textContent=`Đã quét ${data.done}/${data.total} cửa sổ`;}
        if(data.status==='done'){complete=true;status.textContent='Hoàn tất. Đang mở kết quả…';window.location.assign(data.url);}
      }
    }
    if(!complete)throw new Error('Kết nối bị ngắt trước khi xử lý xong. Vui lòng thử lại.');
  }catch(exc){error.textContent=exc.message;error.hidden=false;status.textContent='Chưa hoàn tất xử lý.';}
  finally{button.disabled=false;fields.forEach(el=>el.disabled=false);feedback.querySelector('button').disabled=false;}
});
