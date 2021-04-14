# offical readme
```
generator_cvae is the generator.
classifier_stgcn_real_only is the baseline classifier using only the real 342 gaits.
classifier_stgcn_real_and_synth is the baseline classifier using both real 342 and N synthetic gaits.
clasifier_hybrid is the hybrid classifier using both deep and physiologically-motivated features.
```

# data
```
data/
├── README.md
├── affectiveFeatures.h5
├── affectiveFeatures_ELMD.h5
├── features.h5
├── featuresCVAEGCN_1_2000.h5
├── featuresCVAEGCN_2001_4000.h5
├── features_ELMD.h5
├── labels.h5
├── labels_ELMD.h5
└── labels_edin_locomotion
    ├── ELMD-1.csv
    ├── ELMD-10.csv
    ├── ELMD-2.csv
    ├── ELMD-3.csv
    ├── ELMD-4.csv
    ├── ELMD-5.csv
    ├── ELMD-6.csv
    ├── ELMD-7.csv
    ├── ELMD-8.csv
    └── ELMD-9.csv
```
# set env
```
ImportError: cannot import name 'torchlight' from 'torchlight'
python setup.py install
```
```
No module named 'yaml'
pip install pyyaml
```

```
ImportError: cannot import name 'torchlight' from 'torchlight'
cp -r /STEP/torchlight  [to the  relveatnt dir]
```
# run 
```
cd /home/wsx/STEP/classifier_stgcn_real_and_synth
STEP/classifier_stgcn_real_and_synth

python main.py --train T  --work-dir '/home/wsx/STEP/data' --print-log
```

## file error
```
(step) wsx@nvidiat4-vm:~/step/classifier_stgcn_real_only$ python main.py --train T  --work-dir '/home/wsx/STEP/data' --print-log
Traceback (most recent call last):
  File "/home/wsx/step/classifier_stgcn_real_only/main.py", line 71, in <module>
    data_test, labels_test = loader.load_data(data_path, ftype, coords, joints,
  File "/home/wsx/step/classifier_stgcn_real_only/utils/loader.py", line 17, in load_data
    ff = h5py.File(file_feature, 'r')
  File "/home/wsx/anaconda3/envs/step/lib/python3.9/site-packages/h5py/_hl/files.py", line 406, in __init__
    fid = make_fid(name, mode, userblock_size,
  File "/home/wsx/anaconda3/envs/step/lib/python3.9/site-packages/h5py/_hl/files.py", line 173, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 88, in h5py.h5f.open
OSError: Unable to open file (unable to open file: name = '/home/wsx/step/classifier_stgcn_real_only/../../data/features4DCVAEGCN.h5', errno =           2, error message = 'No such file or directory', flags = 0, o_flags = 0)

```

main.py
```
test_size = 0.1
data, labels,\
    data_train, labels_train,\
    data_test, labels_test = loader.load_data(data_path, ftype, coords, joints,
                                              cycles=cycles, test_size=test_size)
```

loader.py
```
def load_data(_path, _ftype, coords, joints, cycles=3, test_size=0.1):

    file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')
```

solution
```
change the value of "ftype" to '' using files below
features.h5
labels.h5
```

result
```
(step) wsx@nvidiat4-vm:~/step/classifier_stgcn_real_only$ python main.py --train T  --work-dir '/home/wsx/step/data' --print-log
Train set size: 307
Test set size: 35
Number of classes: 4
0.      Sad
1.      Sad
2.      Sad
3.      Sad
4.      Sad
5.      Sad
6.      Sad
7.      Sad
8.      Sad
9.      Sad
10.     Sad
11.     Happy
12.     Sad
13.     Sad
14.     Sad
15.     Happy
16.     Angry
17.     Angry
18.     Sad
19.     Angry
20.     Sad
21.     Sad
22.     Happy
23.     Happy
24.     Sad
25.     Sad
26.     Sad
27.     Sad
28.     Sad
29.     Sad
30.     Sad
31.     Happy
32.     Sad
33.     Sad
34.     Sad
35.     Happy
36.     Sad
37.     Happy
38.     Sad
39.     Happy
40.     Sad
41.     Sad
42.     Sad
43.     Happy
44.     Angry
45.     Angry
46.     Sad
47.     Happy
48.     Sad
49.     Sad
50.     Sad
51.     Happy
52.     Angry
53.     Angry
54.     Sad
55.     Sad
56.     Sad
57.     Sad
58.     Sad
59.     Happy
60.     Sad
61.     Happy
62.     Happy
63.     Sad
64.     Angry
65.     Angry
66.     Angry
67.     Happy
68.     Sad
69.     Angry
70.     Angry
71.     Sad
72.     Sad
73.     Sad
74.     Angry
75.     Sad
76.     Angry
77.     Angry
78.     Sad
79.     Sad
80.     Sad
81.     Happy
82.     Sad
83.     Happy
84.     Angry
85.     Sad
86.     Sad
87.     Angry
88.     Angry
89.     Angry
90.     Sad
91.     Angry
92.     Happy
93.     Happy
94.     Angry
95.     Happy
96.     Sad
97.     Angry
98.     Sad
99.     Happy
100.    Sad
101.    Happy
102.    Happy
103.    Sad
104.    Angry
105.    Happy
106.    Happy
107.    Sad
108.    Sad
109.    Angry
110.    Sad
111.    Sad
112.    Sad
113.    Angry
114.    Angry
115.    Angry
116.    Sad
117.    Angry
118.    Happy
119.    Angry
120.    Angry
121.    Sad
122.    Angry
123.    Angry
124.    Angry
125.    Angry
126.    Angry
127.    Angry
128.    Angry
129.    Angry
130.    Angry
131.    Sad
132.    Angry
133.    Sad
134.    Angry
135.    Angry
136.    Angry
137.    Sad
138.    Happy
139.    Happy
140.    Angry
141.    Angry
142.    Sad
143.    Angry
144.    Angry
145.    Angry
146.    Angry
147.    Angry
148.    Angry
149.    Angry
150.    Angry
151.    Sad
152.    Sad
153.    Sad
154.    Happy
155.    Happy
156.    Sad
157.    Angry
158.    Angry
159.    Angry
160.    Angry
161.    Angry
162.    Angry
163.    Angry
164.    Angry
165.    Angry
166.    Sad
167.    Happy
168.    Angry
169.    Sad
170.    Sad
171.    Sad
172.    Angry
173.    Angry
174.    Angry
175.    Angry
176.    Angry
177.    Angry
178.    Sad
179.    Sad
180.    Sad
181.    Sad
182.    Sad
183.    Angry
184.    Sad
185.    Angry
186.    Angry
187.    Angry
188.    Sad
189.    Sad
190.    Angry
191.    Sad
192.    Angry
193.    Angry
194.    Happy
195.    Angry
196.    Sad
197.    Sad
198.    Angry
199.    Sad
200.    Sad
201.    Happy
202.    Sad
203.    Happy
204.    Sad
205.    Angry
206.    Angry
207.    Angry
208.    Sad
209.    Sad
210.    Angry
211.    Angry
212.    Sad
213.    Sad
214.    Angry
215.    Angry
216.    Angry
217.    Angry
218.    Sad
219.    Sad
220.    Angry
221.    Angry
222.    Angry
223.    Angry
224.    Angry
225.    Sad
226.    Sad
227.    Sad
228.    Happy
229.    Happy
230.    Sad
231.    Happy
232.    Happy
233.    Happy
234.    Angry
235.    Angry
236.    Sad
237.    Happy
238.    Sad
239.    Happy
240.    Happy
241.    Happy
242.    Angry
243.    Sad
244.    Sad
245.    Happy
246.    Angry
247.    Angry
248.    Sad
249.    Happy
250.    Sad
251.    Sad
252.    Sad
253.    Happy
254.    Angry
255.    Sad
256.    Sad
257.    Happy
258.    Sad
259.    Sad
260.    Sad
261.    Happy
262.    Sad
263.    Sad
264.    Sad
265.    Happy
266.    Angry
267.    Happy
268.    Happy
269.    Happy
270.    Sad
271.    Sad
272.    Happy
273.    Happy
274.    Angry
275.    Angry
276.    Sad
277.    Sad
278.    Angry
279.    Angry
280.    Angry
281.    Happy
282.    Sad
283.    Angry
284.    Sad
285.    Sad
286.    Angry
287.    Happy
288.    Sad
289.    Happy
290.    Happy
291.    Sad
292.    Sad
293.    Happy
294.    Angry
295.    Angry
296.    Sad
297.    Happy
298.    Sad
299.    Sad
300.    Sad
301.    Happy
302.    Angry
303.    Angry
304.    Sad
305.    Sad
306.    Angry
307.    Angry
308.    Sad
309.    Happy
310.    Angry
311.    Angry
312.    Angry
313.    Sad
314.    Angry
315.    Sad
316.    Sad
317.    Happy
318.    Sad
319.    Angry
320.    Angry
321.    Happy
322.    Angry
323.    Angry
324.    Sad
325.    Happy
326.    Angry
327.    Angry
328.    Sad
329.    Happy
330.    Angry
331.    Sad
332.    Sad
333.    Happy
334.    Sad
335.    Sad
336.    Happy
337.    Happy
338.    Happy
339.    Happy
340.    Happy
341.    Happy
Done

```


