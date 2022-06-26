# ECG_denoising
Python command line application used to denoise ECG data using wavelet transform, Savitzky-Golay filter and deep neural network.

## Considered noises
<ul>
  <li>baseline wander</li>
  <li>power line interference</li>
  <li>external electromagnetic fields</li>
  <li>random body movements</li>
</ul>

## Wavelet transform
Steps:
<ol>
  <li>Signal loading</li>
  <li>Multilevel signal decomposition</li>
  <li>Signal denoising using SureShrink method</li>
  <li>Inverse Discrete Wavelet Transform</li>
</ol>

Summary:<br>
Mostly Daubechies and Symlets wavelets were tested. Best performance largly depends on signal, however sym10, sym11, sym13 and db13 are the best wavelets for used database. If it comes to decompostion level, best performance shows third decomposition level.

## Savitzky-Golay filter
Best performance shows window length = 2, polynomial degree = 2 and padding method = "interp".





## Databases
<ul>
  <li><a href="https://physionet.org/content/ecgiddb/1.0.0/">ECG-ID Database</a></li>
</ul>


  
