import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node is optional for browser-script regression tests")
class UITests(unittest.TestCase):
    def test_caption_drafts_and_conditioning_survive_redraw(self):
        source = Path(__file__).resolve().parents[1] / "src/videogenerator/static/ui.js"
        program = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
class Node {
  constructor(tag="div") { this.tag=tag; this.children=[]; this.value=""; this.style={}; }
  append(...nodes) {
    this.children.push(...nodes);
    if (this.tag==="select" && !this.value && this.children.length) this.value=this.children[0].value;
  }
  replaceChildren(...nodes) { this.children=[]; if (this.tag==="select") this.value=""; this.append(...nodes); }
  addEventListener() {}
  setAttribute() {}
}
const nodes = new Map();
function get(id) {
  if (!nodes.has(id)) nodes.set(id, new Node(["backend","conditioning"].includes(id) ? "select" : "div"));
  return nodes.get(id);
}
const project = {prompt:"Scene", settings:{width:320,height:180,fps:24},
  assets:[{id:"image-1",kind:"image",name:"garden.png",path:"/tmp/garden.png"}],
  captions:[], tracks:[{id:"storyboard",name:"Storyboard",kind:"video",enabled:true,
    segments:[{id:"scene-1",start:0,duration:1,source_start:0,speed:1,volume:1,fade_in:0,
      fade_out:0,asset_id:null,text:"Scene",prompt:"Scene",duck:"none",role:"source",keyframes:[]}]}]};
const state = {project,token:"unit-test",hardware:{},jobs:{},busy:false,
  backends:[{id:"ltx",kind:"video"}]};
let submitted, saved, sequence=[];
const context = vm.createContext({console,structuredClone,crypto:{randomUUID:()=>"fake-id"},
  document:{getElementById:get,createElement:tag=>new Node(tag),querySelectorAll:()=>[...nodes.values()]},
  setTimeout:fn=>setImmediate(fn),
  fetch:async(route,options)=>{
    sequence.push(route);
    if(route==="/api/job") {
      assert.strictEqual(get("regenerate").disabled,true);
      submitted=JSON.parse(options.body); return {ok:true,json:async()=>({id:"j"})};
    }
    if(route==="/api/project") { saved=JSON.parse(options.body); return {ok:true,json:async()=>project}; }
    if(route.startsWith("/api/upload")) return {ok:true,json:async()=>({})};
    if(submitted) state.jobs.j={status:"done",result:"track"};
    return {ok:true,json:async()=>state};
  }});
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
setImmediate(async()=>{
  try {
    vm.runInContext('inspect(project.tracks[0],project.tracks[0].segments[0])',context);
    get("captions").value='[{"start":0,"duration":1,"text":"Edited"}]';
    get("captions").oninput();
    vm.runInContext("draw()",context);
    assert(get("captions").value.includes("Edited"));
    get("conditioning").value="image-1";
    get("backend").value="ltx";
    get("seed").value="42";
    await get("regenerate").onclick();
    assert.strictEqual(submitted.input_asset_id,"image-1");
    assert.strictEqual(submitted.seed,42);
    assert.strictEqual(get("regenerate").disabled,false);
    get("captions").value='[{"start":0,"duration":1,"text":"Upload draft"}]';
    get("captions").oninput();
    get("upload").files=[{name:"new.png"}];
    sequence=[];
    await get("upload").onchange();
    assert.strictEqual(sequence[0],"/api/project");
    assert.strictEqual(sequence[1],"/api/upload?name=new.png");
    assert.strictEqual(saved.project.captions[0].text,"Upload draft");
  } catch (error) { console.error(error); process.exitCode=1; }
});
"""
        result = subprocess.run(["node", "-e", program, str(source)], text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
