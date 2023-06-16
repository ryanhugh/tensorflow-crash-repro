https://github.com/tensorflow/tfjs/issues/7719

To run: 

```
yarn 
npm run start
```

And the output will look like this:

Note, this is only happening for me on Ubuntu (Linux hostname 5.19.0-41-generic #42~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 18 17:40:00 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)

This does not crash on Mac OSX M2 (arm). 


```
Thread 1 "node" received signal SIGSEGV, Segmentation fault.
0x00007ffff7e64eba in tfnodejs::TFJSBackend::InsertHandle(TFE_TensorHandle*) () from /home/ryanhughes/tensorflow-crash-repro/node_modules/@tensorflow/tfjs-node/lib/napi-v8/tfjs_binding.node
(gdb) bt
#0  0x00007ffff7e64eba in tfnodejs::TFJSBackend::InsertHandle(TFE_TensorHandle*) ()
   from /home/ryanhughes/tensorflow-crash-repro/node_modules/@tensorflow/tfjs-node/lib/napi-v8/tfjs_binding.node
#1  0x00007ffff7e654d6 in tfnodejs::TFJSBackend::GenerateOutputTensorInfo(napi_env__*, TFE_TensorHandle*) ()
   from /home/ryanhughes/tensorflow-crash-repro/node_modules/@tensorflow/tfjs-node/lib/napi-v8/tfjs_binding.node
#2  0x00007ffff7e6801f in tfnodejs::TFJSBackend::ExecuteOp(napi_env__*, napi_value__*, napi_value__*, napi_value__*, napi_value__*) ()
   from /home/ryanhughes/tensorflow-crash-repro/node_modules/@tensorflow/tfjs-node/lib/napi-v8/tfjs_binding.node
#3  0x00007ffff7e6b5f0 in tfnodejs::ExecuteOp(napi_env__*, napi_callback_info__*) ()
   from /home/ryanhughes/tensorflow-crash-repro/node_modules/@tensorflow/tfjs-node/lib/napi-v8/tfjs_binding.node
#4  0x0000000000c2ee99 in v8impl::(anonymous namespace)::FunctionCallbackWrapper::Invoke(v8::FunctionCallbackInfo<v8::Value> const&) ()
#5  0x0000000000f1470f in v8::internal::FunctionCallbackArguments::Call(v8::internal::CallHandlerInfo) ()
#6  0x0000000000f14f7d in v8::internal::MaybeHandle<v8::internal::Object> v8::internal::(anonymous namespace)::HandleApiCallHelper<false>(v8::internal::Isolate*, v8::internal::Handle<v8::internal::HeapObject>, v8::internal::Handle<v8::internal::FunctionTemplateInfo>, v8::internal::Handle<v8::internal::Object>, unsigned long*, int) ()
#7  0x0000000000f15445 in v8::internal::Builtin_HandleApiCall(int, unsigned long*, v8::internal::Isolate*) ()
#8  0x000000000191cdf6 in Builtins_CEntry_Return1_ArgvOnStack_BuiltinExit ()
```
