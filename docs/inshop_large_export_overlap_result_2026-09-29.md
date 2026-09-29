# Exact B32 preparation overlap: modest gain, native forward dominates

Original79963 exited0 and was collected. The sole fit-only serial/overlap diagnostic completed in38.41s with3,785,356KiB maximum processRSS,zero swap and2,858,087,424B peakallocatedCUDA. DGX is idle. Receipt810b3dbd270228432d68bd499615487595be1dde7726070a2baf968b279520bc binds71-file executiond1b7bf0db79c21960d0a4d0c7fec1fcffe120352d0aeaac30c08a50cfdcbfa0d, CPUinput3b62ba20 andnativeGPUqualification9f49b23c. Rawreceipt/log/time underlarge-export-overlap-v1/gpu.

Measured17×B32=544 **In-Shop TRAINfit** images, originalnative256pixels, unchangedretainedcheckpointa7e3d8d, two distinct400-keystrictloadednativecopies, sameheadoutsidefreshseparateFP16scopes, no learning or quality scoring. Eachcopy'swholeforward/headvectors exact; serialvsworkerpixels andvectorsbitidentical; nativewholestate/runtime/codehashes andcallerRNGunchanged. Bothdiagnosticfitvectorfiles deleted. Thesearecopies ofthe savedcheckpoint, notthelostpilotprocess'soriginaltrainedliveinstance. ItsactualGPUlive/reloadedparity evidence remains373/394heldbatches beforetimeout, notcompleteheldqualification.

| TRAINfit544images/17B32batches, two whole encoders | Serial | One pending CPU worker |
|---|---:|---:|
| Whole atomic export wall, verified |10.5292164702s|10.0246452540s|
| Mean CPU image/preprocess preparation, verified |.0203935347s/B32|.0341385336s/B32|
| Mean synchronized H2D+two native forwards+heads+D2H, verified |.5856906600s/B32|.5854912330s/B32|
| Median paired native path, verified |.5857384522s/B32|.5854867152s/B32|

Whole measuredgain is.5045712162s/**4.792106%** acrossthese17batches. This isserial-first/worker-seconddiagnostic execution, notrandomizedpubliclatency orfullheldmeasurement. CPUworkercontention changespreparationtiming; thecriticalpairednativepath isalmostunchanged anddominates. H2D/forward/head/D2H areaggregated; no uniquekernelorweightcastbottleneck isprovenyet. Do not projectthese17batches into a measured300scompletion orclaimqualitygain.

**Decision:** retaincorrectminimalprefetchedwriter adapter, but do **not** allocateanothertrainingpilot on CPUoverlapalone. The originalsingle100 allocation remainsclosedbywhole-jobtimeout; nofinalheldvectors/qualityreceipt orqualityGO/KILL exists. No partialexportcontinuation, larger cap, model/source/rate/depth/threshold change orimmutableprefixsharing.

The next selected execution intervention is **precompute autocast's constant linear/conv weight conversions for the immutable inference checkpoint**, retaining the actualFP32nativecheckpoint/model, F32normalization/probe/positions, sameFP16operators, freshseparatecontexts, wholeindependentforwards andheadoutside. This is a hypothesis about repeatedconversionwork within themeasurednativepath, not a provenunique rootcause. Reuse native/stdlib/Torch support, firstinspectautocastbehavior anddeclarea preciseCPUrole/state/version guard plus actualnativequalification beforeGPU. Anycachemust rejectweightmutation, preserve the original400-keyFP32state outsideexecution, and produce **bitwise** eagerpooled/vector/packedparity; stoponmismatch/nohelp within120s/300s resourcecaps. This targets a useful trainedmodel's inference execution; it doesnot authorize a new trainingallocation or rescueoldpartialheldvectors.

Do not launch a freshCUDAgraphB32probe as thenextintervention: existingSOP TRAINnativeFP16graph evidence alreadyshowed virtuallyunchanged B32p50(214.839v214.862ms) and explicitlyretainedeagerB32. That olderdifferentprecision/checkpoint result isstale andnot matched tothisnewmodel, but suppliesno rationale for anothergenericgraphprobe. Existingopt-inB1graph andearlierclosedlearningmethodsremainintact. No newresearch/architecture/fullsuite/dualcritique job wasneededfor thisread-onlyattribution using the unchanged reviewed HFboundary.

| Dataset/split | Controls / latest complete candidate quality | Current Large pool quality | Training cost | Public latency | Remaining gap / next decisive action |
|---|---|---|---|---|---|
| In-ShopTRAIN-held6354q/6245g/1993products | PriorverifiedDensePE95.0740/76.3916%,Large95.6720/78.6237%; latestS16verified90.6201/66.5162%KILL, nonconcurrent | **Unmeasured**, fixed100exporttimeout | Priorverified.458647s/update/.717697ceiling | Unmeasured | Qualityfloors95.1720176/77.6237120%/pairedlower>0 untested; qualifyconstantAMPconversioncache exactnativeparity andactualcost |
| SOPofficialTEST / In-Shopofficialquery-gallery | Staleexploratory91.7419%vsdatedUNICOM91.2% /95.4823%vs96.7% | No freshconfirmation | No new matchedmeasurement | StaleSOPB1p9919.622vs19.606msFAIL | Fullproductionjointquality/speedgoalACTIVEunmet |

No operator decision isneeded atthischeckpoint. AllCPU/GPU jobs terminal; no trainingjob/review/research launched. ProtectedRustchange unchanged/un-staged. Qualification/test counts are notmodelqualitygains.
