#!/usr/bin/env python3
"""Bounded stdlib admissions; native numerical witnesses run only in CPU phase."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
from tempfile import TemporaryDirectory
from contextlib import contextmanager, redirect_stdout
from types import FunctionType, SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

PATH = Path(__file__).with_name("train_siglip2_compact_ranking.py")
if PATH.exists():
    spec = importlib.util.spec_from_file_location("compact_test_driver", PATH)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
else:
    driver = SimpleNamespace()



# Exact 7a0b984 original nodes for the bounded SmoothAP source inverse.
# Packed only to avoid duplicating thousands of unrelated retained guard lines.
SMOOTH_AP_ORIGINAL_NODES = (
    'c-qZ9i*no6mA`^yW`}@Hk)rD;3KQ-uYs;<2iQ}>5bUG{t1Cf^$DiUA;P_pXLzwde6H!djJO?KA1jmZ0cocB5R@bkCRkC%V__5JJ1'
    'zo#Ew{Pp6O>9^xnQ8bBqb-v!@yJshha=XhG%}JFNH+iu>*<|}-aeZ<->V4b)_Rk;RzWn*Z4CH)2G|JV!SZ;(I<@MWF7w<3LzOV-J'
    'iw=#I7b{VTVxfk6fAQk=y9*eEKS0;dy;_K6-5W=nyl&zT7Z)!-'
    'B>kvYm3LTp_NVW@gKvvXR@VXxMyIFeeVnjZZ$wiTRQ3!?_Pb@)aFb`);wozv*Eqs+__ECxRjEowXHasRi#t8*Z?7-'
    '@Mzc0mmKWL8Mx;fyrNL&sX>ab-)YsXLt7bIQ=<IavwOC}EtimSGPM@EjLS44Zws-LT^OGmjm(yQxd7{bm{8%=VMf2o=Hk|KpsRQh'
    't7kQIs8yZ(syhEy}#JUo7otFg^_O8n2I;vsaCvbQXZ7^DtWwnH5H6p6_+h`cY(D#WrO?ud!ZwdP@vSOLz0YVS4TIGve6wT<zh{oL'
    'Jh4?YLhNgOyXHVWd8JvDc!)&u^o#P5lp#%>?WQ%KBXD?eU#I6y`2*xjq9M&IQ{uph=_DWO$9E;7q&TmEZ?#HMot8KQ)|1ILngr_~'
    'h9Yo@krhp^hg=U)#!o-'
    'gev=CKPRyoWK!{*xzR<|o_II3G%?OjzC>nLBsHuB~^D#SYDimYA;*eg`3zI0qo4tI)x1xsJ<Hv;F<gRk<cZs6deKc791b}%i%EX<'
    'Kr+mmb$3xP=%aClI*&FY&v+86s;Ea{9zwk-FJGu(@JA5Q=))Q*?YS`>RY<7g$aW)C;|WAy6XpPogqS6J`9DHmDYM9aKJ=%P_qRrz'
    'm0myQ?*h#(@svT`Y^=g-'
    'fFi?Ued>*3XY1=v>)cU8FnC@HJqH7sUGxRkGQQ4O~uD`?k?vJwmiu>D~J7b==)SBYI#!LgM^^pvjzHXQL?s1mxQRawDVEQeRw;s!'
    'rJWw~qeEo`ppJpkTo;_*1iU_t@QSlpyc(4gB=tRlN8=0E)iR$3?He-sh?tHi(dc_n1s;6~inaS}}?(a+P5Z(sa1eRJ{m4-'
    'vq5#0_S2*W0W&$2KiGiw$GdQhXANJst#N|FiGTxDQS~@T(HH=k7YIh1HzCdu`xUtrqZ2)~l~kUPROPzkHCz&HYYr9fDyJ<ux>Dyf'
    'LxL8kAd_QSmzt%k<XmT8Io_u;JcsFQ)G=K3sZDR(aFF)l!RBuP-kz-uqjUh%XQK#fyt~mlrR+MhZ8m&HIb#%U?fUwwf&Rolq^1XS'
    '5pB<-S^obXRWj#XYy=&#`sjUM~dQbW<*F>hxM{aFuX~m(^Y*8o445FeTIg8WL9Z5(c_e4g`M%?ac8Jp-GW00LC}4q{M|!YhBSdfC'
    '-D;z6Zp}y6*{=mPJ#Qn;tFP+!<8Soy|_?iG87FxUE=RXFzLf8Z_LKJ6gc95evAj6qiD0@wO}k;f_7}L@qYz>~H4o9tGD(yb{KN)8'
    ';~;HFP{czC|t1SRHY`iUKDlkskjsW!M9k?ATFopow0GHI%v|SV&Pf8AvV%H`#}!E+pz5YG)CnS^&RCWbWG>n{G&VP|H}VBOpkF+r'
    '&L0+8&(iHrwn4%w}!NJ0<*2qN(gIyWf=A(jl|5EJi34pPD|b@>VSS%uwN5BX(;1e7h^FCThxRaqR*+^|7}8Zu<A1f1SQGgqRw*bl'
    '9f8vRW~5Zb07>rvXy{u9gdS3#pGOlD%c=Rsf<(LdBnBJqVw3PotXbf@~3*ib{%!)R!d1u69yr<c5~q=(cA@;8J_FcBbbZ$(EH%ah'
    'ua#p*kk)K%Tx|W*SwAXUOAPl>36Hx5qmCYJyh0Xs$kue&X-'
    '=7k1$GD)w7Zp@5gJHrcv%&OKPC&sdfveM^{;;AiAFkLoH2TcJ!^?JC%W0oBhpV+mO|bHbV%I=0K{<?9y&dk={X6b;bpg~QkN(zM|'
    '1wsCocEq$?sB?aJm$l{a|DM1?FS0MK_(bavM%IQf<<VOQWLpiKEWRT=;MS)bQHwU4FD_i^}&^WX!9KvqXw<J_k?w>rI^_E2J$Qp4'
    '2(lzqlS?^!_Y>6V{qBr;24^rMzBMJ~j)gno`iWNR2{m7IL7^?dbpY?>;v%6EGIfJ}j;g*CoW-'
    '<WTIZg&px{W2E24ThOV!?#E_o)F3;2i|}(C9Q08{i+0j`z)h@e-'
    'rf4Ow$tMpIC(u4({x5+_@NM#*Z2CIJDiWzkatIJl9zfwM;8Z$2OzMBJFp`|Nq*6X8IoB~lLTB+t5NNTFWsH>ehUg7s+SNR1F$Inq'
    'fu9b5tMOd1pRF^hXf8|fh+kamy61@%apaRr2+`O~u=l*B#OOt2*x223kj7q?MBoRSHlv+3MAKjN4G(0mlQ*14(df?(>T_e9@9AS<'
    '^!q5p<4RnZGaPWarrL@NFp{^uR*Xh4@qTLYc=!KRj~e@krePF^1Qr9KooX*#su0%h;q8=*422(I8l($VAOz>!*8{DY3yxm?6noJI'
    '=Lv}8tFAE<a8paXzbpRV_yyDD5xBnc)RS=|&X%<~|b(i;h1G8>$HNsuj1EaL;kif=6}mDe%)?in|@^^sBTg2~&hT*yGRVNsnci?V'
    '_4az%Bzt}>+!-'
    'VX3fx+;vkj!<)pu;Q&;$(1CwE0e{GcORvx$z^BW?qMxyRjhk0f&zmy=QbTOEsJGNs?g|h`mH;KsJN^&v{&kaqR^`YQOW8bj}U--'
    '*@})7Wh#cG)5N*9&TZXV`|wZJnzuus`$kU+T@Rjsy9NIUWUa%0sHYKO;3kJJKE9kr@8AB#C*h0=+NKi+w1J=$!;Cp?Dj^}kXdT$8'
    'c61IZiIHmKkXZ#I6Yl!<9ZNUk#%M#8IvCt3mL)9kKxaPIS%re0*!5T-tnd!79o;IKU_r}Y!y0*7`V&2s(<hE-dIe=mI5%2tkm8~_'
    'QH~$r(Q=71X`troI(u$liA42W$>vl}qIVwsMCNdsBbPInVI%-i85xQhWR0LsHHhqJ+D*~=2rd2_;Ml63rdI-n7e?*@0F*V^N-'
    '8oo62Z#2enGti-2V~<{GVhuU1A{}RiI6IDR-|{INLI@k+-'
    'zfdz6i(nx$>Y(C$g}9PSA;r~P^h+aeta$}H5}>LbG5_L0r<=)a)*6OcT|F@Js<jS?!DYjA@8%Bn1vJ32Fa*`m3OP0PAA@$Pcx+~v'
    's#4`*jz<_<UN&f||!wBez!hbZI_)?%2D=Y2j$K}s0NneOo?p4x@5%r;$YQAybpfEVs}=~D$8tj$t!{2SCQK^M)DJP@1W4Q;c=lGj'
    'yBt)s=gZ7q~%0{?Exnj{_<Y|&~2E865c{oMgSq;2veZ?HgvI8GTk`>_m6jPZDV0hC2do}JFos!66j@`TikJ+v*|K-'
    'vn^0GgrDjFR#RsS(&PqTx`x-'
    'R3nqw0ed*0;+9Uh^3V}bkeJf`A<JmHgPg<p=o^ECyhbTPmr3n5nfTPY&IWqy!f|h!`NjD;gVS%w$nPJsx1RI+trww$Xd&^t*qL5T'
    'y3GYjMLHpquOZui)EbQvcwzhkR8}D0jmz0nWgNGl4gmK*R1W;qh{pT^O~|$n%?H<kaI_U{`}l)42tpwu0BN@I2%h}WgF!gAY_s)Z'
    '?tXN#*<yfhyj{fzGz?+G?%2}6Mz(H2%mB+gneJ9x+&Z%T2laUw&hZ6(((>4yuQwN<WXW5PFhyk0ye8hlGbcG283+w(KSs>5#Ugcw'
    's2^*6gyF%cz`X~rJQQbOjRy&5Jur|&R#9gDi5Q;;?ZDY8Pk)*<~-_>$_7T&YxH@MZg&R<>r^-'
    'Z(&kq79MX%H?nHKzR`B5_zhXvrjvV9k8E}#((K94Bbhn_=E|m`<y1Onn;)JZXWY@>Cpok^iTR!~veN>Hox6`ECRV69H#L$I_rVSI'
    'qJw*rczCdSFvm3)}m@=UG$<b(Dz5e;)gF_wk#W@DHRxO&2bOQ%Z^5V8!kPi~uD5T|KP9HVoy0pSG1KTSJn}!3_%QeTk#Osf;4y#3'
    'ek&Tu6+Oo1fQH8E())W?)05M7A?G4btvqd>6?#)h5e#lN%^Ur6`9{^1s8wWAKf_FaIA;QW~)7+u%+-'
    'c$krtEs1LX24hA`39fR0{cIrKA0alsNRaIR{JXk<^vYA>Sg%x+P&hH3w2slFvUSYpx5CVd$lP$R3cL@RDS|Ua#}Ih%W3(kfdC9Q='
    'lo6S-08atVHRE0OLq*5gYq-NI)qoIuP1`q4!t!$Wqc<xGXGhvMm}<D)CQpMR7#P1}50jeIRJqb&{b}03hjh-'
    '|*Oh1}OSL`EtLoX4lkzw$?a^TUnE;Adf)_U$xX2tiU4cOkjX52`);ATPyj_D#KRN5u0|aPV0k$pJ}9nY*LWl_I0X87RWGQvncc$P'
    'qgK@fxd5A?iyrHT!BnTS#wpf-U`$`0|uprP-~%xsL@%v-'
    'jr7)O1cN_EyGCn>mz0R^&^sf_*6|$l2?&Pb*Ag06Xi~2PbylaBo}7%gKmmQC<234oRH_Y)9hpOwJ^mX)N%*_oo2cM8kkDBmU1DAU'
    'H)CW%qX?J0R}_-'
    'PL7OnfI9tzp>yir*N`c+A>vJ<k+6>x>ybgOHd&qCmDLRiEGY*i$palfKmFlseBSehP~uK2A_1{D?Adn*&~EV5FlZh0plM4)X0FHJ'
    'gUwlcl88p%yZp$Cu^?k0u<S46PBbv7gy=$%S}D&KP_iBSm;w^~hO~a9%w!!(dF2zjipdq0C^m_WNY`eeNcDJ16^brl^n{E}QG`s{'
    'LI<Z3KPwwQ<q`=uNtS@EmRAVJcnQdAu0j6E7ojb89Jd$KpsmbafQyT(76(;UZWnA6<1{w#who9iiMiP_2UW0I25`|Rqq|LZA10+1'
    'zGxjOCCkA10ouFFvvpAdIatIQiP~1d6(4=pkFN0a|DErAAeo(y=SFY?Re8Ua&{#{k599@Qu<~VDfhf3>q1<xlqPpO-HOmk1|5iVb'
    '{Du(c@VS$C=3KNlB}vosKS-`RM6>KDs1W6{>RToR0UA4dRKK^uA#MkMQ4|6+k1pR&U%x#)8x2p-'
    'emEFNMci22%P0}Ef32bS?3f|IKqWVdds+l?07MV(+Dr(X)jm0VwRy4IH|%gj*V0-zol!Sxl-'
    'b8Aw?Z(KbPDGJSVTJ^1vI+KIlfl|D_Ye+oYmrjvZmQ(mO}-'
    'BO&!EPzQzB0WgOi)Ru@BoR{(<KyQW)2l?naeC>ZbSNaXA5CX#a2fCuV|n7RW}7=dmitcWWT!AT0%(>V7-'
    'R&4aJ;(7!Ads$3;Fqs3$44RTeVe4w%VE_F#R;`JVq}Yl^p)&TltHOMajAUe-'
    '5VVZM8gpvaG)2pSslh|n^gpJPNm?X8MYQ)LDGB!3PKwUtcm@}b|2y@-'
    'iyUicwT1<eOVt}zwJBGtT96pS{+`o5y$=X3Mg0R(a9vh4Oix>#@fc${)Kl&}5BIIIJG7@{cc}{P_Ta3qj!HUNXZUwMlXLo5r2cKu'
    'j)TY3xA*`|M9_lf;r2AJhQ8WLKYG(KusQxgwnOF=!&F^u`Fd1C{Myk{-'
    'rrN3YZ(p2?W4Db*NvcQmW0tkSitipJ`ZArB2pe$jGIU7CjWL!!%mS0Ho#L5?RVGA@BlG95QOoGE_4jV=YHH+=2=DOvH*hQh=HXDf'
    'Eggx&}^`T*%sGv!lz?>l^gGyh4;<83VG}}QViXhScnL!qp2lmcB)5bUJ$DIv?D%7WAE|p1q&TG@NE3uT;dadv}KD9b}tWRvsnQAq'
    'Gi(^9ySJqYcdii__`!hpU)@?fwWiaC4wZkgb71M+(TlEg+fOiARGzquPtNa-N-OAC19%_6L}EdNt>oD*EWH0=zty7;RU$ehT_^_f'
    'D)7~0mY~)n_8MjhaAKQx2X&(-'
    'dz~@JqaFD_sv+bVaO>B77wxe&<w0E$uQGZmT%nHXLZ<9LK`Kf<21Sxid1WgqEjQf0}v+hYqQ=sa=0T(yfwamp`#~~rz{n!jHnwv<'
    'YJ;VmUj##Z{W+DPJ@Q(FKhj)N;(Z5_QX^b>UxYpX&iBw)MFMOZucN%B@jpoflux%SxJ}a5UI`xa167z14Qw=??pgqlEDoP4gln~J'
    'p~C{53M+)s$b_zP?K807Fu}gGoYyzs;@FUg}*HGwWu2<L@@Z6|H=`iai$+hV~rdTd$rmg3drPr!TB%CEs*L}w%>rXD+=_$m(^gK6'
    '?+UThTa@!Y0)LACr@Z9m+)Fofjm-'
    '5;EW^+6#~QiwVS6eqru!>%HN|*@V*S}wT8NN)ov!kfV|e`<bqL46dAo#=j%`G&um#9<g1u~tfrwsNhAm5`+%q-'
    '4cXx}g?FiyUsgZlW2<uCfOZoNqgBs<banC>Z{lIFYkKWLK(9erPJ$?gN`>d|Thj&BVxT9M&E;+aZ87S>Qz|<o;4eWg-'
    'AdMTO_v<ojc&%fBpdk`2s7v=$AJ+OpN+_BaTwBVjKY_2Wo30R=ij$)O>g)#IO~Du3Xh-y-'
    'j6<1B8e<gMG~|k_WgI6W{%!;z#bDYiw9myNSLGe7r)wa!F5Gr`a^WKBVk8Fi0L$v!Svdirm+AIOCw+w3aUT)l-'
    'NE3@Bq>&@Vb=0o3<K~T>_x(Dvzl7GC;yHNju8EG+a-l)%w1*fbhJ1xp2pD!taAzluN#bH~SWJ-GaEbi3FCSmuCkbQ%sbOF&MjarK'
    'h8u9PjejAvo6C_^D~pdcHHYU+$<vRLapoo$j*54X(+Zz@VIEC>z5r$TEcjWECGI_KKzw!KZNB*7}tBYxaeM%^!@#ZY*!u3X;RL0+'
    'whLa`0-Qu*=P>8$9CdHPU65%U8EJ?D+MY1>L{8(IX*8x?M1;$)Bn-w#l55@9i72^%(;ZJ6&!J-'
    '<>vWJ$NXDOI1F3!fT|*E8Q_p7EF=T#Kbtw_(^{%jd<3R{@xsAeG(X@jNdIhswWQe2qq*2Klc0+^ccNdtueS{#u8}x34kq5Gr{MnL'
    'k}CZlBp(%5VYDM*@{nwUcgbDzpPiDA@XR+LLNcM0{E~H^y0!IyRzNvX0A>Nhm<aF{pba+PrY2~nWPKm=~E(Kr|^@F;G0EjDFgNw#'
    '_dQhwTOwjG5MHta;jgtkX<YAa!+|DG|XW9J~mG=ZZy`CcR%6>I`u^A(AN~fm{{xMvXo6o$l+;P_xVX0b9l|5Ddio*N4!X7o}OxxS'
    'y2q$sZgo7<NWy{p&ZCv%am3y_1-'
    '6JY|N(?cYw*2Siv$^@l+0>1D5n+sAMo*X4}8Tv*{dB6tlO;_EObD2^L*Nu~h|N(OUvq?tuq_v`5~76biEY9$vMcNvVtW+W}QH>ly'
    '*qj}*g2ulAVaIaHtZ%OKN^MIIC0ZfYo44Q>u*=hqb~P&V9zL^&!@j<~7vbs-krGK}}xG}So-S+X#;kZ=m$!pI>X?2Q;tSdCF<(}-'
    '?IR}XnO&le_v!$bjF(5efarks7pZYPap+uh(V0MQ?)KDKEkXY85CbUlLNh?f#KO<;+;BFEigTq%-'
    '%_)Z*ITa|aM9_~BHj*hm&$C}K%)>F^t{Ql)#kxpAWYPQ_wjr&rUgg>eVcC$vIeqTt2BftGV<|J2K&2F=Nlaam5vTIT;AAd^w;;_|'
    '`L4^ZCFM=8J2N@Z|?2_$M@<x5mV-'
    '03md{9uZ(zm!Qrp1FfBS_!m@#woybDRz5INOa96n_uf&EtBbzD*_RBe<=Q1&)opJ63FGZigec#&d$MEwxWYpXwK~-'
    'XleNJ|@EkjDJyIUqfO82mg4PLpmdXNXNSr2*X)*{}Rq%0dR01gXjg~%A_YHz!N>dNe^uZ)x=T&0A_Kiva4-ZRO7<3-'
    'hm<#>qefBh8~${H|MA?H-qn+d@oFiE=nm;bl9A%2Y!xCE|5#5J6(-'
    'T7oieJd>`_+n!uAz_|Q#kDEX`q<)Jt4wqaN?Gq7tvj7AS^!e(f(ClOm17~f@VM&yUcbXgV<#I%bPCN;Icmxa`^KN<iTMzoF5_oy-'
    '+0l=OCfsw*BR{n=Z%KtlS-'
    '=hYvL&tHfY5g(E9JHVXD6|gZcdb~Pdy0H1Kt0@}Ty8;JSIE3$uRVt~&yGOrFuC|4nTL1rRnchpJZ^kGm)7F*K*X7CzLU#8j8_1Q='
    'L#6lAA!aeQk;_+#oyo&M$%5JaCnNoyG*^H^VZyo1AXO<{m_p_=fC@v$*4hEn}25<@f0Fl?bK1Wq#@^)*1LUyl1WwVF;^T=)msd3?'
    '?wR=@^$|7hfa$7*hPM|X`gs#z$^WqoAsqhUuA-'
    '!z8JDIL<{AREtCfg{Z8JPiEpDYda#ZfvHCWSfZ^)R16WQrSd_c_G!CT?nvz<eg2z<Id?+%9dg{c217-G$H`(@TnT;P6Q0?SIR-PO'
    'TF~GEF^4wG$pp!NY`)rQeqkWEvt$uHuBewX+(LT{fp;Ed4v_R)Gb7}AWa!xKg`ZFEj%kY=v3NLL>#`+jf#!=o3f99|s1DS1yBcpS'
    'TJ0N)}Xyy48n?ciTm#0<^*U)mIi7k&BFqTux_Zra5vvVH53I=Xi$TDw2^AD&-'
    'Hml>7{f<>{3b2_3jC%@qV9l>KSu<5_DV$82YsvC#>+VXIK#vC{9YpoN<-G703-'
    'C#r3HC}*i3Z4*cJ?yX5FHZ!TSE_a<&MX(pUB#2LPjo8rjTV~SL3e76Ox%(!RX!oDH_`KQ7J)<HH5L$Aic=*v0<8>ZmdHNtj?@q$G'
    'ERNCEj;nk%{|p8shl5ENR~MW5EZ<5q0Fy+F#|ok{EY)bHzjVt;FGO2b?d5x`m?f$6u=M+RrFb8FgmW9em?;lyB(&ffG#D9P)uVHn'
    '2~q-T^r`?T`Tdp-!&oDF?V$trRs*A|MdRo^rhLp_WoJd8+U`Ni33Mo<h;pgM3}2ESOkZAiEU`-SvQ*(0m<>nmrqXFokY9MS-'
    'N@v`dxjG@#mkj84n9M-Jt4`f+zsR1L=V-KdxJ7%WL~jB-ADOHa=>f&rLPc+?e2nCwTb_w*(yf9nN(D=EMKEPmX|J4#q)8xzMJdA7'
    'I_r?O%h{%gVy%&{x~1axUTYDgfZKnv4>{TR5%whLu_z*VB(&4&Ff^UyC0gnb?ZMMoD+Z13KS)#*4vCnypTwNjKhcFxjIR(5bWqAb'
    'o}WM5~@^kB(P;f4&0uuDnt3cVhY>O0Yl^9Yi^GmucQ+i{{?o_%tZ_8oMMjjGZQx>x(%811Atl=%$VZy3W|Q=PgHox4Ded;LdJy82'
    '1pf{j!j$17BKct2_;&o?mjQx^L7h0N6r`8IL$yLLHwaCX5ohr5CCy6M%J<XhyQ*3^V@99kK~20Fq(&$1n)imhIk<xQOgtAG=Qs@2'
    'aVs5+hOjFz!2=^2+^!?BLo?q-fN_Dh|&Xt{?&p0;|}K4mGf28f&F66BYq_>3WKdMFbp>o}#M9<G)A-'
    'F8)Cy5n)jLub#8J)6#FH}k-~+gS-'
    'f1x~ZuHRnvCk7@lcnKev4*Px%`*p`@Hu8y|*x`F;6zT8F1@24HO#Q8d+D_GoHRxo1ayDqtM;PI+B38l%l4I#QiM3w3Zgu9->f-'
    'Y4`Ua_A+IY%SEs8*>wUYPa7V&<3@jUL?UTuGi2cF@Y5<feaF0!VN0y`=cG0A6oa7qklLz#+;|<x$R#%=Cxtjh5-'
    'ug`yUGwYfw9&>j_M-'
    '8>8KS&*+7g=On7fc;RY8!WO!MG+=YZ!TKp^uj8=1@U|7=ftGk99DHJTcD}qY#}O#Gd|AYV$MF``>@p$G?J<?k%%?W6Z*v;JVL8bb'
    '5^}IdXQ-E>6jOa?X9-iEhn^-F+!?-'
    '1;Qzc{(u!(9ldfKz4(^KI#Bguvg?fPpk4iV4}Xg%hPB$f%fxnYP*W&X5%xIxp%=7vpL$WS8)Yyga)j?nL%$J$`rjuVnIbbO>ob+H'
    'VP^5}dpHy}aNv_r7|1*SK`_1r#@XicWKaY9ukdLSDo_djJ`^4}z(-&paoV$=wGv`XqE`d+15gAA%X~W-'
    'S?yGA%)b69r)WYsDqC=bBWgfOoQ%{tx>(dml>H)*{{>`BYRa48OwIq<6KS=U#Qx#3A+JaNno>9e2CpM4?ll~AWlZoP+j^~Uo=NhI'
    'cC!!+k;F*9^@Z0`iOoJHGza|{k9J0*%)^)a8|L?x_aht587fA|SB@J%U64OO(tf9lqBbs71D<~no*`8|<-'
    '|*ld6W1`QeO4p<P`K))I3E;EaF*-'
    '47v|LI)0!4Ul!VLH(OX=+*k;2vX<({xIR_3Ua@{Q>0`$l#&ysZ4yRJXjx`F|)pXcoKXl>0{pj%lDLbN`r0GXs;rMXk118}m3GBES'
    'kJ{o`N0QqbO*a%$AUm<YEBc=fU>jT}1JGXMB#<0IMZ}_8HXyBVbt(&x;`M_ueEoKiUY|~x4@V`=DK+;~K3Lw1#S?80tonJ45Lv{|'
    'Pt@w&ZZ7yZY{CpB`Gj@s`c7n9HCeDNf(IMEB!C>@(|-'
    'K7ggRAu3LN)@JN45w7`(%J%gF!r8XZ1PQkl*R?9#&n%DN>9*7tBjRaqeOQnmR4iBmwlmhXwaEP00xMDoD&_s-'
    'v;RVdginSS~dhaGoX82TNSBHQcik3E3@NnEdCVND4Fe<5u6<Rs)>kx^6MjlD%<@P12~Hh}!|N1(|vcvQaBrj*yXnNusq&wqTaWhY'
    '!?4Tp9k;%Q?4`pd{jxaJ*jSMHGmIY&lkEjQmIiAF|eUVw+nDPrI#oJAz$qxYHHqj^eHMOR`UV26(3g)=ijsrCC3mPjH^0+c)>;`?'
    'liAr^bHi1AlvIk9iC;{oySkn_Vn6@IvXwAfDutHW|X$UGWWyQH@eU=On8GL`FT2Z7j;4%o)6!2?HWTPG{Av}T<mx0MtQntL%9L?Q'
    '<CC&n?8fwF|f?TaTgemXHZ8M@Z3ab7oe5~L(S5%B3mLsVxb8l)F0bn`R+Fu|BG-THN54Qo#{!+r2(#4{zhw4fPxZ%|5_YxsEf1${'
    '`3Ijdzg{X(eX4x!W`YN+n~AqM_{Mk7tM@@sVFV~Es6gvYl?Z`$h8f?K~cdE9)eZP6B=Ja-_ywjAo5ERtZ^i&AO(bt;n+B_}v|2&#'
    'r7o%4;mOXM2dgZ-{)-{hNmK)+-'
    '_WD@IBepf_4{Vb}gtkQgyYF)r)lLRP8{26kP%QJE%4n*PhThOGSjYTv6M|)ZWl0e1Nk@!T16)?EUUs+KgB*ieK_cu9U(sHObG6cZ'
    'K0d@9E;FVXD=c5;RqQXKQ4JNrd{u;^A5`x;86fC}(aDUcq_7qe01ri_z+uUM?z@>c~Um)KcmK_%?`-'
    'DxiC|DkNOfDVH+Joh0^gB9QO$T{@O@ya5x$)1tb1aPn14TFKFaNMNxr6Div`MB)6@8-'
    '<^GJSRvhpaRuYMzBY*<DVYmRUXo01ND_oqFq{62nK9`*e|<{pQL_&-'
    '*n68qp6oY?=#ow1aDsml4yov{X5{(YZu0P&?aI}~Wt3Z|erOhGriJ?Niu1St4tT^57oe!HuIDT-'
    'T;H0izg^_O?k7nkYBcQ2=x7d_;M000jF8W>@SzYrjQtqo1(cWIFoQg_D`();>4pq)bhZy){_4@}?}'
)
SMOOTH_AP_ORIGINAL_SHA256 = '18a83be27e434ecc649304cf89affeb7a7ad9089ab7274381acc6b42ecf88c11'
SMOOTH_AP_NODE_SHA256 = {'AUTHORITY_SCHEMA': 'c8ce894a446a95bbe55d91a7da9c4ad501b530f23a0890691451d643c86ae6b8', 'BUNDLE_SCHEMA': '7fb0fa5a54e3c6656e673e5827a336b34d22ea371bdbae1ec170502693b3039f', 'INFERENCE_SCHEMA': '2ef7b39b1f944ecf28407868a0a4b38687b76fa270e6998bd4a48cdb4d8a89d6', 'RECIPE': '70113290169f127385276a62a03300f91878349730d4f867536c51ae6a3a881c', 'SCHEMA': 'c296494aa481af4bbc15e7759b0c68c0ecf29d880d4ffc8e943463ddd8c2f714', 'check_launch': 'c63f5699e97360d949cd21deaafd7feb264706f5f59c658a984b81a5b3ef5b53', 'check_payload': 'a1e612bb93303430b5b1f832cc61174bf68f4dc66c317b618bfca5584720e931', 'check_steps': '3dcc5e276768ec48ea65495512c96765b970b1e49babeca38d9f979dd8a1d0e1', 'check_terminal_record': '229a5dbd0f273b37161ec0c2609fe8871f7c4fd89b6c2da75860cdc42c597b48', 'cpu_gradients': '5f13180bddbde1d66def88541b56e6bfb0c4475a2b1ccf73797b737b063c31cf', 'cpu_witnesses': 'ce9c0205609c179eef07545d59a1a999d5807203381b407f973752ccbb99f9fd', 'fresh': '650608ae9456a230b3b9737809ccb0b45b37057a5d669461cc24e37b7884a1c5', 'gpu_run': 'c70acf3b090677b863a8118a6118513598ee676553128e259e31437e14b08487', 'identity': 'ed07044e70ba918891d73a9c09bb0e9ec685c5a577dbcc79021951d0696674db', 'integrity': 'dea97f5e0d2a34f9d44cdc63002cecf562dd2cca8b13be4930f60e70b4d47d06', 'loss_denominators': 'ce86de123238eee428db8c62ff8ed4c09c2cde5ad6fbe52a85139623802a78e5', 'loss_terms': '7d2dee77280be380508ee3495536d866132e2799fe3a39a4d1d6137fca20310d', 'update': '376d51f4188bd6700c958de1ebb33bc770fee9da3ad7a56f3fb932a6061ce533', 'check_cpu_gradient': '4c466add6fcdf6af5ba9de2723281f35833ec7be8ca5e9d5373008ddea3d8cc8', 'check_ranking_bank': 'b6214203d0f994a7d5258101053ea480323f593b78814c4e85fd0c30ee0e1a28', 'json_sha256': 'e6afa190880b91cfdb1e70946e6618bc5792b89e4b5e2bb8d72e8e549b865d98', 'ranking_bank': 'bc2b9bb03c0e4437e68e2b3e67b34dd0a8bbef6b7c2fbeee7ee613197e14905c', 'ranking_membership': 'c56dd67441cc81a47e8fcfb71032e4c751b7d5b70f2b6df19ca6236b9301813f', 'smooth_ap_terms': 'a90d81810d4295691ff1829215a07b9ab1793cd040ad92530d6bffaefa1eb698'}




IMAGE_ANCHOR_ORIGINAL_NODES = (
    'c-qB1i*nmWlD{(Rt1A);_>d!OZ8}UH*Rg9|-gsTHvsYJK777A}6f6?p0ia~X<$u5Kp7&rtO3B`x99!hfpl7D1-`#_JeX{uV(_jDo%iB-'
    '?m3(~l*VjKUp3GRMjm_DhDK^_;|8$g9yM3Cqqh?oC?e!?#kGAQd%&teb=YuCPd-9iG-'
    '~aTt*A_6R;W(tLLz!>6g7o(No7ca*e*ek_=C>V(SCngB^D@&Qe|i1t?T6P0NPL3FzYZGC^JXw(+oEZQA78)z>0=bLL0#Qp<K_3yp24?ln>G!{h'
    'V$w4d5n{mn=NmvlG>ie|16pQ865U`+KSOnaqv~zX4eSp0>12utgckc`6aa67W_^x;2&>a|0gWKf!Ee)QKna0o|Ng1;4TM?!78?Un(K5gx|THQ`'
    'Q>!wjmXk%TH}zX(~IX*cZxJmcX!bB_3Ui%)8c1bG;dSADr{TIrd9NfFjDMs17rAHlto*lTbhB_V%uXEe$6*EZ<?Yip?h!z2gDlK%LvfSX#Xs$s'
    'yc_IwwyJG9h<NrKsn>nXy8Fg%NsyGg2~b{FA&x6Fki2Wtl(vP{*npsSygkEZntb-HL$*0-n?WVi8dA=)D%z8-udubzT5LUZ4WhvImXj-'
    'L?TbKYuec$%`(1kdCp++sw`l~?9)rO<GU+f1Hon6LjzN>4=-'
    '6+)w^_C{Kki$BC)0s?v(MVH{%ZQCpHADMS`<nuj}eJUa~9L^h?IS0Qj5}=2%r18$j2V6}xo9U(!wqr9}={DB62_-v*#S@|FQz-'
    '153fx87nJ*l2#(;@$^(igi&pEgT2?%jE?F>@eh+m+UUBccb*sUgLxroCCD&(&naNhw{+y+&?*<=GCEfAiw(X>j+p>p(<w^ULJ~)vo)t9fVIE*@'
    'cmQV9c=N?R$1D#EH4@)X@3HNETRmM1;C!10ryuquP-j1PqM097n{k|VGUBDne6K-'
    '1B$Hb$u+ENLL6SK3tmrlJO#FPK_f8<k^}rq+O*#AmhEf4Pk~f`s@ZqqgaISC){(+Jks+L}>L&_VS2dt6pIoKc4Ss;mYTp(+KtMhChJwmSyobm+'
    'DcxXY*-'
    'et|56PxZbL0gCkn=U`_6#>M5mhu3+Y|)cz&w(_64`QH?1wloW}BrHWUEL#Q}h2m6g3}ic{#K?*?bO?jsb7tBL)nLsHe{^2P+)#xv5IoFqA_ge8'
    'fgNW9|K(BWNTygj_VNC|d<uv_cCbsonqhJMY;k`+<pOT#B52Ik*K9M1Vi2^@pJW?%`m=AXOj2c*@SyEFD-tB0~x91_7092#{j7-'
    'd1T#hyX@vug68RM&YW)0pYM480`YXpW$HF@IfrZBmM?zZ|iES4?IC0Qv>jPI~`h@`d!`G?7<+t4o%WKoo)c+Zps~e!d8$*B~O99+hkicjo0J?i'
    'J~z65GCwKz`VlBA4fu9r^mK_1cgVj@}Oz{D2jme2n|Gw=R{JYbNmb)fkQFF!c@OIY}+DHVl7F*<g4nY%ancN=%hl6?0o=O6SOo*mwg0}pVL9_@'
    't`%!s>g60y-'
    'T;yrl)Cnz4KPzS=o{2aB;}rxWHZdMJ;l~z*?9_k6kFvl@fT=o=L;E>!dAslNc*A;a@<uptNh)r~1HyK>ElJ<aD}i5=jmg1j0Ysqvw~&W(!v7Hm'
    'O1TXQ<Kq89k~Utcx$8uE{b$tH4oupw*e8F?Lt9B`8`@+N5{~T6=(gFA20!UQNsQjjOe&X^7@{%o<R<ASJWvfWp$&Ut+9cYwRJd<!Pz{5DGBBBa'
    'i^mxE10SPY~NkebEG66fbK~I<Oo-3Z8I{CydV!Igp(U3)t|XTqgYj-A>S=D7MU!D`?6AeY82Xyj-D%fP-Ny!OzSp)(5cij_7C@?rUjZQ#(T}T}'
    '|x>KT&!R`tZ9^j2(n%F6%t(hy~O80BZopMd7Ik4mla(F|<fEd{-'
    'j&$Quo1kz8>=GdIil(_HnWYv6Lz5d?zzo`z=xgafPPjS!+@mAb_uF54QC$bTLNzuYa!EtHhDlceARARG|a3)>zBpRo%dIIu$w8<gr~A3<9{;}}'
    '}pr$~FkGL?ii5gshi4Qwy>IaiTr5li`os{^xrhi(%OY;@)+Zz1B@+#%QHuwehrE+Sa(`Q=P}fxVoIhLsC~f7$a4y|5#^Ag77kClxD*;o}>A-'
    'wf#_z6ykEcRgmxnv2<)n(v!1&FERZa(r6tDR`RrW0UyTo0)=fvohw3o+R>lLiI%b@QDr8!um|qIrB$gn}V(xy~jW?B;yEs<;#u85oP8aLUg0hZ'
    '={A;N!Xz1VP5|cWh(5fef6_V4-&OWjt@M{C|4u<&GMjWuFScP`3U|bhE<F?Y0DN2I?0bw)GL%EPg$0qup7mSnZwuWT#B_+RVY$t-'
    'Lpa+w4AO2c$YF(>i{t<Y+Q#i+TBD4#V3GD1#ol}E;J+q$$A-'
    '~5+*UiXoDC!*%b}iUW3S(CTRH~nI{5yu94z(4#h99FDNz|v;B42a1crLE~eOt5{c`&+7b4E<ahfD&fsH#JoX;reBkAC*M1RTmlUMgrGoHffPl~'
    '-yJ<GjOCO!iln{a<3quwD3FsMwZ;gxvpQO}lud7_IL2ON`G1({DqJFSoOjn0G<6uy4i|k&mkv<P;sgbi*9Tid1UF!Y*3A*7<VXI@r(*Z4FMagY'
    '}F-'
    'aHv#l`df3DA@V3P~`SC5+=YU|S0mqX?1XZj7CM%xT}JLKHM@U1Ti)1;)sw<qggJOM#7mnCnCj1zNy(8xY&B%K0{_?n)`zpjU>LlB!NKK!Su6^5'
    'CV%K?p;Z(ZO;vy=emJP^_U9M&*3Z%bfi95LgkzPn1$f)bI@b8Tegb)J{Zz!LQA3n&g^qbASsQtjQ%k04ddg;-'
    'Hu#yGxyQ)xrdN1HL7u$KAoXI2{hbdA3^rbTVL=?0E{F27I_Fu7r$zjwZ_V>BZUEQ`j0&$hyD~v53^MyX$JpM+XpSj9fBo^K_#SBZ7ZAEEfAx0S'
    'p439mq=)7Q$Q%T@L(?US)p#hfp&&Qfsay*PO(c(i}(!XoJZGtI+d6nrmo?S>3jKriMJsal<A!D9dG_sa%j|%NVg20-'
    '0%>MUE6}>iD9re8rgYU}YJHZ{Ggx^+#8XDl&B(nNBa*wS#_7LgTHxtul%u<9H?Y1V|ya<aULeRdLJXGFLt#1g^V0-'
    'DZLI^@4uBqX~l3U?3b;1fp8xsHAhUa3?XNWWGuavr<<5NFoOV{hVaIO*f502Hn2yIxXK{qg}~2=xDh`KOm2lCK{tdx<gBUI9N_cFVfL^_4V@N8'
    '%VKJa3ViuVhI9q1QVpRiB@wH4T<S2jz_9stO=bCn;&HR!ysg^X)+eHpC>YE;geI&Zwo~m1gyvCR`3pan#g1yCTo2CrzIsyNE!^1x1z*Qa&_OT<'
    'DMR$#bpKpgIXGOx>&CSGw4}u;Gjv!?5hIfoH1ZDljeM2sL)@|u&}StujoOqtBfe{66!S+1TkS4^iWOMORX4NpTBi=k4itmJRJ^hQ_M_)J@A>_H'
    'H5fJw=nCDP8ccA?vo0>29is5hgM8z2K3A+pF$s{6AKgynJrE31-'
    'HB*FH53N8QzTyPgqIyepDVwPaT~h3fc6OW+PFBQAnwb1w8>6#S)60IM8l@gM$14lsItcJj@T_@T^VYgw13P5+G4=U&}vr0<^BgPMX7215)AOp5'
    '=*Y9V6xlYyJR6OfsG7zyAU@EvuFo;(b+uAu$+4kAr!&O`GJds&B~ZNkj%$9>?rr`r>l-{DjC-CKOWamav6%CO{~g$?LJvzT??-'
    'S{7N87&S=L4Blpl_{4FtSHJ$WI7xbB{ZOD8Jeohdf?dx$APU7u=i|h@43Wzyy>$D<Y=NkZ6KPizhu=^Rp1T3}?m9>sE-'
    'A`kLISl8Lj2e(@d?OM#6yIAqM&@oh6Vs&25l1JL6zZ<c$to)Fjy0xcefLf&}^~DfmK(?KX_g!p|8O%DY6iDdQN5*4lI)BvB<=g7?vVJrbWJR1}'
    'ODoclIF8_IS|*78)Qg!11kfUn*E0_<oz-pH3%rSyMZH3589bhhUI6&x>?ZRv@RcVM^As)o>N{yCZmJDNq?6+i%6*r^My+*($ap35T5|`9_*-'
    'ps?q-63VscgJNRK6N#D73HR|pXOb86pT`HD4hga7@k8obP~OC3B(M4p>HomwIU!UHTZBcc-G37P8n%BesO<dDHkoyp3lEw`vroS)-'
    'oBq+o=>KiFR(C&PM|43v&jRulWXAE2|LZgQmkSZ13O;<ysr#Beot#am4=+@o%p%{#fKr1p*21_P8CJDKeWl_0HgqT8E?4rFnUX^3a~ltmY=K4N'
    'k9uQC+&osRdxjoi|($bc8X+LCYDLF+KodDQ1CAzd)ab@ihn-'
    'AKW|LRSJ!D_?zbwp_b(ZqIHo^y${J26ugqGoK5L}5W-MSxFGd7A+p5m4?MNtoAD5--Ujt0kp5Vu)z$Rkk9mr}14inGKV#%~7n{^GmL}gsgF;b='
    'm^ek;^sf*lDs({=Ws%BAUv;x8!!@|*Yjcgbc%#dS4u?(w_3ez7)a#OS(3-'
    '&uUx1Q9A7&BmDW?tCscz{}u4GHH_M;58l%UCTa7LjhyXdqT%`aAW=8MuTpg1{zJk~-'
    '_MERLzx>xPq!CsN<jn5tZZ3+d+n2n|aCV<tg;Cmvq)h=}x#@Oqc10zr>rV}&MBRl8c^@6}Syx$03rJB)MHbTmd;!9+w-'
    'RP)i92G*c+j4^u`^@%iv{BgQR{W*lG`o@aGQ3GpN%QzSmr}v$$T21cDeMWDMi6og>5eQFe5YzhVP}GjVLxL%>_7bJU82=d(&;@Ga78DzbkNve!'
    'y}&yj`C()ZAL;=mu>LSurr@`CSwKG$Vqhtv<OD{IbC@i<9!3H+)>paj*sMD?(<;8{6l53{yu7dwDP*ACeSYaB(N*%5Kp(K2R~SE3c`<volDs57'
    '+BQy)#UatM-DW_$eCUIReSu_~pYzc(Xp$c<QH_fFAFE1L76uUyOmsC*z<RQn=Rw$GqOPRtNZ`SM#@-'
    'Dh<3`}=cx?*GT%zXL%5Lth`#{t5cAR6j4WAW=qD&;#7h@KaT6F61HbOKGBQ-q^f#)mmN;$M*X3=#1=nGIskdm=+6gPb--Jv1*k-'
    '~0tWDz|zIxbNJU^=S`lSnXs>ePT4(K1136Elw{AFZC@Q?WHAPeeyp3CK=X{gHyS>gfV>M1U~o+gai^&2Abo6Zp(nNZx{H-'
    'pbkqT`_Mj%aj4e*5^dI?w;~Ir8ti$O^^)%jd@C2c4}k(m(-*%yR^{7I{YEUjAO7osTzMyeoUX%r~4SVjW=U+GlG<8UhL-'
    'Q5&I9dBzfBcS2g$t*u|gvISWHrTcw8GG9*?0GwXLabqUM7hqjg)%QVrd9~AfrM6&WxqTJc>%p2bp8w{5r4SlBFe2!JwpV@}UeCc#5T}#4JiZu='
    'W)0_nd$b($1jxRydGVZ@Hi;vMins3wHRi3iXGd3KNlpKl^TuPjOUTLiv#XYO|yo#dZD<Innd7dsaB<WsSTlQSQU}k9wYbW4d@%ySolVe$^TD85'
    '4SaVbu-'
    '$6`DY@EWG76HmoqZKd3jH@w9uZT@Na=^Pr><rAr=%LW1fB_p1AqnU~sYHH{6q}|sJoJbi<^k*&P_BWYwWRl$qb<F@2db$MXk1}H9eX2==*0(UF'
    '9Xa72CcV%s)coJ;q~9xdlBx%TI~r|ZHt;}=^5+dZB+ry0jVfOt3xXtQkX0|@Ua2e7Tlaq%gd&!hszf)#-L)HzZk1QLuT$+F;U-XY0dAm8O{g(-'
    '+9cQuH^5SeIK)nU~VONTxykL&|qVHuGhjWHMRt@M9I~5!>{TzWLKK%cZZMLkbo37vZQ2`dW0A#K{p;mQ7bBkR^g&(GgTxCi?#6?bl<JvraBrkn'
    ';|mm(!+@o$w%vMy99V?)@Ez*#9DMxyPKe}Rr3IE*MP=b4uOL@CSUyNd&G-7OmVPITziYQ3O@T2%VITj31a9jHI{w9@w#e&Qf~NgLBV5-'
    '^mYdj2sO_*(yp;^Dd-idY+cZHRs0?;P;K(6_}D_r6&rxQi-'
    'Z0VDcm>X3<Ochlcv}eTS_A<F9KLTZGrWM#u>0o_+T%ZS<}c_F}cIy2J~RYoG6%eZJB;kMBOjmpM~!6L4VL$z|77}?hJz<sv=gkQM|)cUC*RL2Q'
    'Ba5%e!z5d+2_Z0;o{sX@%h8!iA&u(t;W$2&?acm|mCae<ZYHcghs(2l<P^c>;IBW`-t%2O7S)H52ZYdtc4FKsD<aI>BtJcLh#u(zh#&9eNx{7-'
    'c7nh^Rl$d-'
    'A&0EOc|~=&9OQwdB+6)D<l;?wbnmYiq*}@ask;*_B_^jl_c{MP(qzVPO~+RJnz6m;@1`6BGYnve{)2C~O8;d_igO>mtv2*}3DSGY+w^z|;!WSE'
    '|5JU-Duj+!S=Sg;+KJIv1G1MHfe-Zk^3LBhieYJd(ONZ24(oPk{Zr+5uUuQ=sGo6gZk1RXyIN<pJMw08djY$_`%0r1IRFNhdXHx{0v(<!3({Kq'
    '43AJ?5~C($&?LBD3MADplE~suA;(Hr{CH+tl4*v3TAaZSV$G_=eZ}8lE>#(*5^bVb`D7pV>;>3;G@ed}2)r%AY!#kP7%uvMwsTrtmIx!dGO>=0'
    'x;kFwFO0n@`w!V92>1eZ`abHr}^`E)Y<Cs97a>Ht8)7&mUXUMZrrCl6r=6xZF)(EE>388nh?U@Ce4zgJiep=t-'
    '9JvYIJ`lw<tUi>)C&G!&}k+48jErw1x-LRntb)&mCjv3+8Cx6|OPLe%{pHo&vd7$s3CBKJ&^6%&R2eWeHw-'
    'gKbyAW+4IPZpU|r!47>?#%ezz3F+Y+E>*#BCFO|*3bjHcihoQG(j0)=PSC8pleVx#~^Ywvx-uy--'
    '}g|{%3R{4s>E7zMHX{?p<M_%LY!Qr{&EEr!34#QcM4ME-'
    'T0GI|~TU8_R_|bpawU>oREcdvv&8#imD)cK0rGe|6L1Gv?IwYl}icLC;nB?K<E7nCl85fL;hv!|l{!dsTBY$`<Gg3^P|+>iYz4h4;)6qy7wv_!'
    'nllzhcw3vfiVFzCRX;C;K$J!3DarDaqn*$kejT;3TMc1f9BicvQJ59YPB31o{Wvji~CC<s&Cree>5Z$$2Kp*}5Q!mcL~$hN`QBmUeH!wYIEJR9'
    'G7eysyqWJn!++YYp{St#ugii5poOa_}11(Y5y81Jf`&)y^|fxP}*3d#j3PotRjEEu5YybQ?}}o}GyWcc31*3}%rl6>^%(U`z2rr$mUM#CoCAU_'
    '}m<35h9h>cJ%;=5+}DzShI5p2YhqC=dZhU7;|7=Ty+nYHEho#LhLWBb7@Jt6Fkz)Q1v+fPt~7{vKm^`bcU}*TUfw56;QM=x!PykTplB<i5+?;*'
    'VhBInPmMn65;F`;Xy2@tQG=z51T2@S$@4#ShHulNLLko6Wn$5opyT;87^S`GG{IXqLKgNwck1(YOKaTVp6fC6(wtG`@GH3y4phT=>5v-'
    'AZKc6^nVfZjm94nZ$-k)kSWY9&1`O!ot~BuZDNlaVFK!BPMT$L9B{RQ{+O-'
    '!R%q7L;rtH)S@P`GSod8Qi1=6rbzKofMtUo<LmB*M;suJ1?^`$JZ0WApx?Bpw}3|zG|Fi|b}}_E##FqaABKZ;)X&QDIaw54@ka2Gd%DE2mI-LO'
    'Oo03=810Nr$0swN7{j)cksVQ0-'
    'UKnDi@`kO*(#&@dUj@5q#F_^%_Q!0D6J&E^~E00!tGZ=I`tytLC>WpJE_KNc_S3xoG_Aa7o(SpQOgf=`QlTNVWlvje)ELOXVqPqEW+-'
    'X<(oA3z$JB6q6|~LMTX~#j@B4Ra-GP94YkgDRK>5_iC_5Zy|pN;9fOM1E%JRQi{6snDqnpG6tF<=QH|}QU-KpeK|VrS%B|t(Zu_DGnetqT$i{)'
    'S(^d&kB^TmWVvSyk)EE3)m`J-'
    'c8_<?I>?Bs?N5wiSedAzniq*!F?7WnVR<BMrPULgpaOk}c#HVg3x3AdhQH55IF0xXoS#v_+LMk99pRfE&0jJZws-y2Z0mv=+xe*yyk-Wa6Ok}b'
    'yJtf%SC1lD$?ys3R(XQgNq>;MB@ki>R?NjWD0sjl?gKeycT+V|n%Wj2<F%hL{2nt2h>M?u4C+u;W=ZRd8%Rlx{&bI{K?UA)Trm)$nZ_qdPJJ;f'
    '1xR+-j88N0`Ls6%rUOqeQ9y~GfcXN}g?OV6zdOg^wruIe~9j+*<w`N_73}qI`w{y*i-'
    'L2fTCb+XR1@xOYEh3hP(;{h6;iULVjF&h%l=LAPYO(3X%|wF#5u_y^IOVq}u>B=W9hYU%Pmh*R6$cXr9~n`cbROx@vHvg`x4sm&WQ{MbdxU_#N'
    '9q)CdyDE+^Ny57;qawjrzH;GU83m0J!p|rmg7TPY@0E?e~ZK>_9w&T0&l>M!Kkk5s!ob^q76A)ZxYF%E)X<?u>`?ukwaU!`wm?xx)T<hM{HeyW'
    'Os?_%o`AmiWjnBu~KiUv?^$@!4I1>Hw7?QKGD0HU`-'
    'l8*O!Jtk6uv<SE00P2ohwe^zq$^54Iol>s^G|0ZtM<FDzcX4Dl$t5!6q9i+9a(`)D3v{T#PAGhrD8EKV>S(HuPz{};PRbSl=pWBA|%txnek(Ys'
    'IYF<{mMCDL&u@qaxKL2sgoW2HWW8FO5bp4WKJI`8mI>~bJxzZ?{cc~m)&)HbFTNU0Q;PzSR9Ln|jX)Vk*(3UN9%l|@#eiZoH(sr)Jt^*!#d5^_'
    'Y9{~|wAX9AwBH<$k!(wVftW9Gkybf#HDya06;kTo7~FIAy(o`X(ol;$%H{S^L9Od#kE4}6>-'
    'cKZgjGrtw_=fSJL|NLR`>QnOTho2UoUgNtn&w%8<0Wo4q4d19V7lB-i6yGIfTFTfUz7_D$T-*F89b<U@$+!Ojprrd<'
)
IMAGE_ANCHOR_ORIGINAL_SHA256 = '549c67075e3c3891930d6dd88e5107baca896d289a9717d4f14f6da414bad3c5'
IMAGE_ANCHOR_NODE_SHA256 = {'SCHEMA': '4218e5fed8dc8ccf25947498c501093963bd45c9ac6e600ae1c5ea3ffa3f71cd', 'AUTHORITY_SCHEMA': 'd0267ef02058481201845e0749ff1f32e9956cdf3d6509371df5243090f1784f', 'INFERENCE_SCHEMA': 'a5ffb29526ff524d9a6f350ca5d2a4c8626ba7812d3a6f22d7262c5c790197d1', 'BUNDLE_SCHEMA': 'a44ccb0ecea15bb47bc5c2dd53ae2590a3091fc044214bbe707cecdc9ce8ebbf', 'ANCHOR_ENDPOINT': 'cd7e9da091388fe766f95d82ee3c8b4fcd1a7793e76a97199ae57c778cabfc75', 'RECIPE': 'c661bea4e6031fe8fc4b88b38b840c1a806a3c3d2c1702adf7db2471a5a09456', 'loss_terms': '051c24eef6cb3c07f2e15365a24c15708ad0e9ff707a09fd44c8aea163c5e19f', 'authenticate_anchor_endpoint': 'c40e8e35e917df89f54579a723405a0d30664fd5df951a5d6681ecc6ec3fcf39', 'anchor_displacement_witness': 'ad35be405d58e040fb7b0ed48b83d37deec1f73f99757ccf26daca027a8ab5d9', 'cpu_gradients': '8fd57518120408b8de8d0fd93e697c0836909ed2af1212e8c131cec457107fae', 'update': '793d55489d4454f5471c7268481f529bd223c84eb0cdd054e2a3b0f1c84d578c', 'check_steps': 'c4ef99bf95d4592f9f8d96795bfd84ad78ca591eae13d5040aef3f81e56ed052', 'cpu_witnesses': 'aca1bec7408e17521a6d28095ff4257f32a7a7143146e8047ffe044affad80d4', 'check_cpu_gradient': '6b853cd435af926813bfd591218c90d98e5ff4062c8a8e7538f84df042194327', 'check_terminal_record': '72f681866bd458e555c35e01439ed405c149264c3685e6f86e549d6dd879f833'}
IMAGE_ANCHOR_TEST_BASE_AST_SHA256 = 'a8259716541ea82e38da84c5eececde611019ed4b25d315ae65492b9d594a602'
IMAGE_ANCHOR_BASE_AST_SHA256 = '9b87c006e5f46a55ecdb79b015bad185161c5b7ad30fcd0d3c54026dceec5586'








CURRENT_GALLERY_ORIGINAL_NODES = (
    'c-qB1{d3#4lK%=W_k%>2lI1jU6XmIMerYnz<TcH->AShn({N}KvKdpPj->3Qnf&j!yZ8n{%FcD}+$NR?V6j;2Hx|s7M;Cv5`1#l0-hBA4<oB09'
    'zxw6k(X#i5t=Ob3uF7Knc$U?>eVTQ%VwYa=Sz2Y+bu(*sb=_Uh(*3MV5AbMqdp3C#Kl=M0?|%BnD+Aa0{s5qthbk|*1p4OP>sP<MdiT<T>pyw`'
    '=%U*4hF6(_{@bgUZ{EKGnBoIK`Z8%b&)dn8l||c4e}DDrr{ANPO`7@+AD(~z^eKGH%Cv1cJ~&$}p2awSbyf1NuISn0_+J&%Ka;~gPdm~3F?PO8'
    'yX+c)J%KN~B5P{(<m?=t+!p*!Bk<2Rul^GdaNu=KT2$#}$&)JG5#06UVzP<tmi9W`i>5VAdUn3pcs;VTOdIU-c=6=f!kr?`)7>33eK|R~`03&o'
    'M4ET0#tNUU<)cmXl_*l|aRGDqTvSC@q$SP38?o%U3%}u44R70`uAq66)m7KjC40Xv(@S1%Ua%~!@&c)bpX#arNbJL!b`@GLVLMsf6j!j6k`cXF'
    'R@Y4q-(g65*s)VKg;`JdBAR$Jr_~K?7EViG(q`AP%QU;WOPd^^@a?w93SM<*FIZZZY+tvqnp@t!V80U|UA$KlJU)5rL+|)*&zrP6G#qA_FV2wU'
    'Jk73&4wE#?_`c)0y{Z>%$9I>!0p`rgLkm-}_b*sgH@mbf{^ZjSky!8y*U9+8n{fx!7Rv!_kv%Qg+ot}LSFpP3<^|)Q0DKPA0iH3$Ba(`hV#BP+'
    'Vd+Kp*<QB+C`eo~;G$dJv}x%f)51dYLy2pj=qa{E({`{K?C<AK7*L0yguP%u@7*jtbk{gx2Kxrjc4>RlvO{%fdG7BUPxJcFIgnqz|6>N?tVor!'
    'D_$LninA@JEdbnKzyJO*t`3%X=;|zOJC+wM3V=U>AeLaq-5!LQGtd-u&YLGso}B`VZi}na%fl8lNPD_(>J0d|ZceWOuT#<o#kSzh>5iwM1YFQr'
    'P3$nqkP%M1w7KFP+c$il0;_-wu<yhU14R(mnW8;wVI=IXY9|iZ)(x`ispi@5SiSFx9T1?Id<CVZ73G7~qfm<Wo+Qv45J~FGkDNX|T2_a$j32>5'
    'vYRB^ACjvk%~4)p!klkezhQb66AMO5K@`F6E#M_(F_BNMi~STk#_Vb>mE<N8V;cV7hoa%rl2=o!k*!vsMHwgy9Adzr`g{EJe6qm~AKSW;52mt9'
    'ghT8tVyyeT=Li~^6CoEZE2>U`7Ej>`Ba1LR{H-^3!T!d?V?;$xzn<Iz!y>@HDg49G1ogupRU5*1%uduSJy^g{Qwi?|0aa`YkYcti>$D?6fY^1{'
    '^P=6N;#U2D&Dag}c7fr~bh2ytBo?X^e@pePH#|X!R2^`CdktE>`%V4X?9QO;4s9|xoqhnsy{dNb37>-Qs(1>@pi9cSZM{b>kSM?y$XG4X`NtKI'
    '9?69iMIx~u;enzPkmXzH5m@vX6-KBzN}zj9^A}Ky$Utm)M&z2E;b&+F?2qXd7J%KM?21I`!$fGTwlAMU{b-AVTRJoVq!i3MDVlu*!_R1k_qfAa'
    'detz*V-R(VY<ilGgSX*g-@|NV93qFJjtK73PYTHy12<z4Jrq%#D<$w~c9WKu+oUUan;2&{;h#XwpyF%UhvvY8K!(^4RCZdniE;Ikyx2<r`v^#1'
    'P})TJCE$a&BeAn`2YR;ZzSs-rM)E+CcFDz}KMVQkXXnXP3Ep}MtI7`<+HQXmPs}FU;uAO~<O8Aq<Cu5wUzl1VyDPd1=)?q(+KFfR3fPsl0~~?5'
    '=HNd}C+N9OAU4U1OzLs83dAV$8~(YSl6h>Q>&0)scrgSM-Y<C^DCegdjfR8Ik+*8zT;ZX&0_^tGqMGH&gRV#qS18v!_nv#-qz{hHaoxe5$~lmM'
    '0i&2fw1s?Z)&q8F2Mb!7HB^TkZ;H&rMp`C=hJ4^zETP(h3h`u$K@~Vv6HnU_cGQNLDQZNKfugnwu(vkg%ZWra4#Jk6$^i7^N?Z8>DwG^%(XV~}'
    'cl-lD`<@<x>WMAlvoMa@7RnVz!!ug!j1UIt0x@gbH3NVI(dS$6NO$NFPABWd?0GudZoZs9`8pv!8UY;*5IsB@!cw+vwXzJIqQZcAw|j?LhbJT1'
    '2tz};?X&0rFul8Bf6{4@$m*{_fRY<B5F-hgJS0JHPZ~FXr<lQ!N$kwaq^rx-8J|6kS$f&3Z|Bi{rzJU7^E$JMz4?Yo1gQHedsH=61Z%=LZG?Qy'
    'I;7vVu3Ibr5kHDGi9HdYLpCfhu*Yq7L}U+9#(R9wpqCT0?mk-vb&3nm4?*q=Sem_i^$t6O4p-e5R6OSR_<cuo_y~In&l`gJni-U4GI^kx%AnY@'
    '*m=?J%QWLCb&NLMelpp({Aah>)@#D6d|wwLmZIz-atNqyST>Bw(k91L3hNu;ZwNowB(JR~!@<{vDESf|UFgsEgKx0R(9h0LAE}&I$U@pEtPL$l'
    'j4s~2{Q1}4l2`A3djIR2cOQrs-ToJj*m?b(<|7ji5`M(Fbs>x{d<2Si4;*m!^AmbQdK3~tbbsrtsQMURjqyw<LiM~44;|xfjC&UT(ngms#<x*)'
    'L?yY~IJ{Os9b?0xT^C08n^lqKI4c`00wyN;3G<kPF-&)8Dq^rlDs_k<Ek$j@EIQdkuWc%R#Y`MG#cY3_wqScHRKU!fiVL<)y(7y6&dqLLHy!)E'
    'z*O!#JSN!#QCyWLz^*B4y-S6R%ABd#f6+{{iB|gPG^WZYqui~(k$MJ^KBRCANJx^xyX!jFIEbaml%{MG&!S<lU=->@1J-F@mqqqj;z*yTglgpA'
    's?)?O;VK544NuTNEfP5hJTP)Vi{yE7Taev!z(0BNY<L1p5r6_pFuO0b(Cf66Bt}7%{cenc8p})WQ<1R(5r&u;fP!MmrOyVB`KJOO0%5Kb-4%EN'
    'l^{w;or4I}ca_K?W2?-yCv}r%rOGD+law-(qV=>sX*AQDM`_a)TX+h+a=r(nLiQ~LR-A<<I>n*7XPBpf--WQf=yoP$o!zv_H7`N2SFpf_Qf=U#'
    '!f4R(Aw`3!WWCoEZRiEynMKk<Tb1KRntpe%FHVEQaLDVbX>>Yyf|BfcdXqHp;ikA0>A+{8#m*LwpPZaLhNTgQYztI8;_!mqUDqX_9l*jc^0@Gs'
    'r*MQC5%SX$vDnuNQ~rr(2lDTQQeKI!VDG(#aSb~7L!?<5t+mpcYeniyVol~4k3}TH#ypF5JR7l^mA8ILp^m+9yM<0vyjTMt!A4oG2~FjKv?B0{'
    'Qpijnjme$@r-6`0A&Q4+PEP9en}59e-BqKCO&y1+*NQ>Pu*;<BR^8Sa9jsx0rS$|zp|<35g`QPMGvhi}4iO61MNdDoK>Kz<kMCK6;52wL2pEB?'
    '7DxOvf=>BuV5XNm?ibyx6aQf%$rKHU{I*Q5T89n#ZGAkg{Ea5N(r>~^?PqX~Bx{bq>18l2MIg4|6gllwo4HdBiRoi)k5!AUwCS+;Nwz;smVtDh'
    'W@EwoIpD?j0d~syZJ~IBfDL3A6ud*8CbCunPp^IZr!^&NNa_rlx28Npa{0Ma+r1c{#T5nugH{@Cy1-Y0nG9eX*l1ERG7hrS&w;MOemUL3`M_aA'
    '4|ZJ@lz=RuT|-3>6NSMJHAKDC%FYa^=*i`0H2Mh^E@0;_#ZnFM2R>3TL<!WigjsjA!^nAdn>6qc-$S^6=){Dk!@!(!6y~XVStMbQ+1ea0<C3p%'
    '$%=eBCpp>BCRsH}<GU`;GLFWp;msI1!ym;ll&%)%45aMvxI>zA!I1z)36&B^4%{D*!JvczFHamv2}@KsuC=K}`;lxxha~DmOqUXPCqb^xPujy}'
    '3wq_?a>J2n9XDu1xc>%jP_kI)KYxLsS9M1s^{%ckc{zz52J^B^+vKipZpd#+L{W^~j^&fZ^Yi7iBT7#>VaW5P$U>l|6x=PnF0aKV`;KSVDMX)b'
    'VhkzC!+H^N&7}OXmw)_pag_DQf2t4Uf}Q=~ng%`Zgg6u{+8$;o2ym>914TukoXs;y#*M?#t%*|c8*0+CSV6N~I~xzur@g9$Mju%GIA}Boib|`K'
    'i({qL)%fr*)C-NR7bLzB>D3Mk!%R|~p%T9a*QLlp%O4>9()VP<a!I#XeBi1C%T}ROBI7$9l!dYzhfbQTp!jGm&I;1^W%~K}S#)3abZnT=@$s_|'
    '4AR$mkzQ3bD57kdl7nqMxQvD)7CLVQHo|TDv)KErwtlwU#CEdbu#-I9O0N!V`U)|LXqVMSF|p)RNtw_N4>5UflIQfl$LOAs2+831CiR^xZ({PJ'
    'H^ZCsf8Z4z(V9jq!n)ILKMH>f%fA*}cJ_nKW<BA;ji%o0!*3UF-Yw40P8a9Tv6zu|pe?|3$qlzp^&$3?*<3@VSfw{6_PIXDzVZe6XTkw(7D}eK'
    '<J$r(8)i|aR{v<6D~f7==#r}gumV;UU>Suq7z3(K<=~v>Rw%4IP9Tf&t|*>mmmshh5o>P8RHx-awJODFKbv9(gMaE!nt)FgA^!CU|9fppM|5Q?'
    '){xXy`S}II{Z6YZ>;WOwDIHm|4&2iQd99rb7&6Qe+0d?TvTHk4ir?o|rP{Xu6OAM0c0ou)&pXi63?eF?8)V7%CZ~1_l0=2C&L&cQCYVuL)LNGX'
    'qfP;>F;$O6spSUPY7U5_ZyVV#DBB_XhLRmtF)27|WA&tjj|KZJTUjG@Ii~4wVP?LtpKF^V+Q<&W+O1en(3EPQrWf;^cPWIrQOqRm)+(4mVZIn}'
    'nbA?!gk`$}>_1Eh8<fKTs)YPvVJlSRENxz=#0)Y)ehqZ3DVkrA+KYOcQynhGihwXunCTV!a=S!Dh%$VO5={3IzQ*t@_RnqX3Zb<i1D;;CNm<<R'
    '>4gX#5^@ocZ>LWq&y_HRTSQ)*{p5r`;yHV>n=ElfUp+N|m|<PCLT(v8imA8TmUm|MMNIcEh*HrqtR7`mARs~vv=%Uy64ZBM@PI$mq<6%TcZn)S'
    'H{#<hG*+x8Zhmam_<OUKvu_3rI4DZYEm0(-5Alfwi5&8|II5W}hn)b$(qhNJosjhv>7B+)-e91Dfe#ad88D1>Vq}hsj^{`XYT^chBsJ<!V)skV'
    '=;Rw1{$fEY2sTdF{c9~Z8K4cRrgUGhm61Cc5-T8}{;3z0Rw6NtoCp`+w1(%{VKTQN^<vyDm{83<vfV^mNyM1FjfRY~5=_Am^K_5?;1nj-dSz&+'
    'kYhChzaASRCH_0xc{FuE$1o&|28)+Wbm(e)Fvq{9M5qgNBs=iQu;>0-c#WX=NAVwnDk%Rju%@7D7*&jP1n6NnTGOpb<IK@j|1`x-ieQ)WJehiw'
    'iH2h)(C3HUp~S#n7>TC1Jje)@oN_bSV|2%T(6K|T*ji)cy67V+z`{YGZ{sn5d(KP2$Ce!28#ONwP`UtwLKTT&T_=_~+d?1<_KY3!P}7*8*lYdr'
    'odFtC<+Mb~8;oL$UV-KS-%8RE_Bw>CLj#0^;j-Q~WlRk`@HP&-jm6b7Fmhl{zgSj{6bPmL{};ur*6K{XL;&VrS-xwP@ZuI>)v@oh<J}?STo{K1'
    'O)(7fH&nPW^eYLB!t0X#Q92DG2drEG6f}#EKaS-=sPM(LiTu-^f-73W0eK*3|HcDuABd83K^wVbH?|&1XscpFQKo3!5tH?cUTOD09B!wLZs$4H'
    'Pax66+Nrh&uh41NR#*4jMo5`yM8(-tOEz&O(Kumm)gWCrtVgbwPi3VLefxf+yB^yuK)ya)mK;#R(jbG`kG(mbEjNQ?^(yu?mBq3xC3h;>NS%dR'
    'm7Z=To-6oRybA#XA>x%LfkIHN;QD?+fnAF7DxkKmR*uC&kZFvrh&AzC?)f_AjVb%SloD~gh<r(zr+^PG5Yr^^jF*Gs)8*!<Y0{)v50v8*1&K21'
    'xxf3-!=RdjXWZ#4-t560?Ti<vI$B_j%V>vL$$);-<+)BwMg-^wm;rDrfxcbkJ6@)3G?)lSG1V9Kl799gB!ikMj%y-`15<anrIHZ|2Ddm``|tav'
    '2Eze*X0|U6?M&aR*7R~y@&q`x?u4)Uh_0V+EcC6S=I3_wj%5wf{p&+`=EV7dB4B%zHNy`ZHVc6JA&@q(ccVYz`++5VzZuY`HaNYuFpOtIFs6pA'
    'GezrYOpy*8?*!=eU?`rs4PX|BXS#nN>T7pVPd#fP#LTXZ`ZC#<ZCaGBCbyhDE$RKue7p&W6K(OJ6kAi^?L>C;CL-motUG>vh7*8*;Bjx|Wt!cz'
    'Vy1n`v2L_0Ds?y9T>rOMZmuTA*5^bWnt3^)W0{$etjyVY&?X#7$DEx9;<d3n%r|CVL`GDHkN8r7Db;(#j8nB_9?rC>AH3C1j`-`jdCddGQhBjk'
    'oki?FLW!4uPB!xo!*L`Y9+xVsfCS}K#RFibaw%gGpuUp08sM$9rZP>m83t8s8L_M!N}RhK&j|4n3<J;&dFUgp=A*s?XqF+W)^sT(y^#gW(6`j}'
    '-&QQxL2l$`Gme7fWn6z@kdN`;VpXQQ%RFTtmuxx{E{E8`wWRsSjY+P=16X|AMA0|~A`t~LPu4I=_agb^%mo5wP*YSp0{4=Ct}C>7>w-vI#6pJ{'
    '!kVqZgIA=aq{cCtX%(OnO@#4U%($7O_KMi4#}4@76n4Vi9-~K3!F;zONCJ9LD<v|B$BE6;aMzJ6kO#1r3UaNvj)%Nq+T39%<F(z>5{5uiX2%V@'
    'ya{L_!>0EhM$rH>iox>i)1Hs^@9do@w7~o0r+AZGT+EV(R{BD@x*&j|0BX_t(8;qRm@GT+xdGV~-0V-stF~^Y>*vqsU}Bs<c!-Lw<jRRxqoLi>'
    '&V?^#IA6@xXE8&|O}}IIeaxP0?vc-Sv7Cbio8!0vuSA!nz?_0ul(b6AD}LFep)lRt$amzp3n@t9(MZi`pHn4x!Un3S6IK5KhDPx+Q_(T#nF##{'
    'Pu8sN=B*}p7#3o)K0jE;cY$R;+;&?>9-Fnn&9>r+g>=%mi=gx5RweNTWB4<QgAolH-@Fjms4@NGPd}hu+-^#O^+BVDw0nhjaXesfQH@#P#4d8)'
    'Y$i@-8lG`VdT2kK&Z8#*iMycXs=^xdK($w`j;=4XUa<k__k-vd|MffrMO5*mEp|moC7{z#$OZDBb|8mS6T-wM><hM{&$V!`@l&c-ya{fyWX@@|'
    '_4+pYO?5B+YloJhcMd^&Fd)I~txEF1=SXF8)ieBp5BsT_xA5g{2xRyossbomd+Cx8r0}J68?~Uh3BnqBgN?>ZwLcKr*yURV`(9CQaGt<RYnFyV'
    'gBu!twaik)TJ8&Vyf?fodtOhlsL^{zk6NVfWEiJ(97vdCM?8vXxZLvS`(&1(H>HSVlAr?qa;_9R?`9buKK9<2rMj_y?0qpK;O}uYT+7CNG9nM+'
    '?9un>3?uDa5?#sw(YZRBNUyAC>O|R0kjL+pC6*OX+Uc_@p!72}sftefrApw>w9Y|(gI7}At43HOCJY!BK7-?Imw}^}r<xin1HLYD;ECb)+r+{T'
    'Rv4<U#7OE(UR;S_5mO$*xS7A5x7@`{%7nTayXy7yD-`Hep2KAe0Kt7jpIhp$>1m4ZsD@9yy3Pv41;IdO@_Gl#Vw-|~O29`$Z?bOYyR<ssTRdSj'
    'l>yo*Fj>&MvX*iD^iNgwQ2n4`HhmdR)Rp;%ChAt4y29g^%z3?`?W*Z_i^X`ax7n0n%AelXm=B1DAHUl5IMO2tq&+m7sgQZJ954_f%WEq`A{$zz'
    'F18ibiYt<+ImBLj2)|C*c4B0unSH^X_&VQrlRgk|XwYAhyx*p`JUoAFO&0~<<083dD!a?o1o~nO<CUNeB$FN>wYrmR77YVsK7(>>?y1Q>{_e$A'
    'mjNCMWztKYw)}WUB~56d>&6;jXdl}<rdPWN&MNfiw}=7mjixB+S#f5>BwI0Y4}0kN`QD2TRDB1o2;s;gbFPs}egoF&HxC{tTbD7_^)nRJ2208Z'
    'U=Q{iokTN~0d}^b>*BipUrP)UR|~7S$MUUI#rmBGS{``|nQFFy@%SgS)QR|RCeZ@pY8hSYeWa>b-gtK|>o~Q=GIz3)_jx||5D3p3OT?XGZjrQi'
    'B|G{S&%(c3q<fIh;k^gri(AdSWX^r<P@%BO>F6@)hg?Gs0rb+-8gB27t#_AcNheAAV*1imv-+O6dvkZ`=wE*ZB{dhOyN|JH+gPs`$Gi|;{!R92'
    'c7q7IvniG2FUVX#Si(+F#XrXQ&2-c%D&s?z@d)`R{e`G22kQq;wE5b5ZKGi4rDSL8eK;66Szc`K4(QUq0BX_Fx>*~HBnB_r>Zt0-?ONS$@Q!VG'
    '-tlX<8tT|R+c4lGZ{yIAgRcM?#dYu<AR1=xp6XKENP?!}`=HBSPIdT_lb$K`CKT0pav~7!K^=J!h{e)$L(jYjWG&uqo(NTySZ_C<Y%mZs8C4kt'
    '9!s(6`e>{fN+`8F93$TUItwa1nG%GJu2EgVIF<6X9<@_zV&?@MBUe)HcO69?$l9jRJx&G+%*CzUIUa6|94zZAPB`M;wR_OJe~1()4ne4ts*oqs'
    'KZ5t@jH9{&s*6)EtD`sWW9B|o1xJ<A{;`hUX|dv!S-dq;zW8o6bSHrzZ>%#9{RR&EdIVMW2OzvB)hr%MTPq=MoWc)`S#WJLiBB|d!Spwn(NU+J'
    '_-4u$U((#4g)f{UjlZN0fF#JOL<tmOh@<<MNkW@5<j6fg!fG(R7vVAhuO4jK3xQSF5l6izd)6I4G-slimE?}u-G-_v{~s#|(8O7#69Y&0g8uhC'
    'yT_pb%LYBie^g>R61I8LZIA7VYlWBgNn&-~on8WNO)!+8_1LHUq2fw?g9M{T?4&c$*70-XmgC9OdV-JA(`SyYtc)PhMb#h#?Sw7nM~~$3JjFgG'
    '?`gs6CP<lHA)@r$t<tYACnrWk`o}Y9@^m&Dc7QjZ#Xp_#^?v}Y9)Kn'
)
CURRENT_GALLERY_ORIGINAL_SHA256 = '989b0ea79828555a88484406566d6d2d9e9ac2aa03b2511ca5db46445ff5ff41'
CURRENT_GALLERY_NODE_SHA256 = {'SCHEMA': '5bf925cba82c5d7ab5666d4dc9d1aec568a8b1b13ef4c3ffbdf4239571c8b142', 'AUTHORITY_SCHEMA': '7c01bf4343dab1263fcbbbf959ead2b0b87e2f0b30958cd7304e32e295993ffc', 'INFERENCE_SCHEMA': 'c101dfc275d33985c04870b7d55d770b25bdd09b64f64d36912713066d483c32', 'BUNDLE_SCHEMA': 'd6bf92ff37477b6bc9cb1967fb285e6b43a297e8f2876510a75253aef8f2e859', 'RECIPE': '6cdd22f7573f9124acd5799d760d8bc18ca88bee2879024796be6cbc041f32b1', 'ranking_gallery': '5a4f329d0dbee85f1e5b0edd9ff6d6cd2000e921139046a883575ecaf2fdf762', 'authenticate_active_objective': '84c40f0be0fc3586d150f6d736269bf40cc7f497b99f23572466431db875464d', 'loss_terms': 'dbbc041853c44aa9e6c9a51e42f68ae1514a5620e2f3e6c5073386b57bc8f300', 'cpu_gradients': '2c5d10239771d1d2663741affa9f0c5891a57e90dd270d2f64fa491e091a8145', 'cpu_witnesses': '21f97ff18d31f1615476ea9de37e2e60ba49ebcf47709a24b6ba24c94ef0c64e', 'check_cpu_gradient': '6d091ad32e6d29f8d6b8437442df1a00b404d011b48d0bea31edf4b0669cca1b', 'check_terminal_record': 'fd57ff78bd01a96769c70e1dda424a5eb2278bdd9bf91e31f290c5ebafae11df'}
CURRENT_GALLERY_BASE_AST_SHA256 = '7c29058aabd5045705cf32e009114def4bdb87cf235908b3fee7244cd3e39dc3'
CURRENT_GALLERY_TEST_ORIGINAL_NODES = (
    'c-qAq+iu&&@>d9*2a*oWNNSv>4)owAZez4@g4iu^APWQ~t|ZnJ$?{^y5%lk8W_FicUM1UUPk~f2IXkmEGdp)}2k)XVOWZ|v&C)EjGx)!gv1H?i'
    'Zr~*BFSf{1Cw7y}hrgF@kTS;&JtvI9r62kka~5&##7Pt_2d4v%EwyK0Fj5vQ&C{Du6aOvNY;mhiG<<Hmv2WZA=#h$l5hdQuz$_O9ZXPbymgdNX'
    '{DGl>q*0zMn58WPca;Lz_GZ8z-3(5(o59lmn&k=nv9s_MLDk=s$mVW{d%W62QMNw&Onfv<8y~AQ=We!G16<&znUT2RiW#G^rTzHYRDd>YGiM2I'
    '$xHk@me>Gz>xU~xyjc6Ofru@RPFd-DEX?>(HY{659&T75;SUVX2h-Bj(@Iv{4BTYH54k{%h=Q8|K{7_DMnC>Ed_A&Cg2jI~Aoe?7@(zT_7lOoH'
    'Aj}dcgiqq%^k#5&GniH1rfWAAucxE&8%rC#9?oX%gu;Bo5`TeuBxOQh5ToQ2vImilPhR7D{1fQP2Qx)*Tz7={L9aHh!n3P)AI?9X(N@=PT0h)V'
    'c?CdR3J4|@wOt$}P(ORXr}VvkNO?Mow&Bt8ZHSwKb7)7}>sQ(`O0-b(LwTrQE`C41+I4Kq%|S~NL2-*B%GUglFN;_9UuhJI!xk*Q8O#O2ou%S&'
    '+CVypCybxmiKD<@JSc+T4-CP4(%vY}{Ehz&lt)|(VmiJ7O9p0P9xx{bEphpq@$l7~VF%r15`AN#bLXdi6sqW6y?TQP|IXb2kCHH#MEeWcFQOD>'
    'poT;b1p=A`G(n+4kMG9rAb9TvfeJXbNyn0C!BQ}j%)NCI_;Tyd3C(b5_33>(jYYJH1GrqClV&VNXu`6{J=Z~#3x*iDS3M6=CJ5MIv~Y3t{2}8v'
    'vx|Fvc!!`q>ICVGluVT-au;R7?SwF;XkFW}t`gVtLE*Vk*p~MSKM^U}KCb>5zd)sVRFqCA5UQ+rY6v*9m2!O_8-DFaw!C@xmrZ&G{t|E(YWEv{'
    '>(K3{aL8gf@|55RxiKl}ph7iX+vCH~gKzac5MLVs+H45{E;LV_&J6)YMH-+N{wmzSH+*+|;C427koj<_Qy&=yj(-8RA$<h9AWR6En_y5;U_tNC'
    'FAtcMG~XBj3ys)39+AO|(clR|gXNd2!1kMur5Hph(XoeQ43sxbW)W{gmo~U@Q}96$dpMcTQYQt0!hMXw)L~yibrAkK$=M?XS``;V)08?=FAv7I'
    '@fS%ndhM)&Xifo?moM-JSGTq!R^o+I+{e}V`TMI`m(jt%ERKAxA65U6gVA|eS{^X!S<{Iu3MbM|V4jCwz&eogm*S(tOL=<;_~DWzECeT%X2@@K'
    'E$eq*ei-MV6uFyltG?|zw0cmhclVVYSE81`b04Gzaf^X`Kn<mnX5b+*kY^CNo)jj&U|)TF3{lM5QNRHW1{k0M8wef(C%O;01%>zRdIa7wO77u^'
    'P6P=9*mH%5Q2fFlmb_Ge%7>T@Ea5r8c`Rn3M`opyE@upo^;Z=2%QXwUUGV9ne<05QAvoOH#ck@WS>OSB%sj|?mC;Qn86hm#-3FI&U>^3tZB9sz'
    'u6kF9<O4F*f%`!rMCNW{2uyh1L)+@2ws8aC=Z3sAs++Q;WGLKpxcGU-$+m)^k!oIJvmWQ#d5l)X*xvFRP8a+ZZ@1K_U1QC(;*4jYa#MffuB06='
    't)R$@jFm}fR~&5%idhQ@fea)n%*U+Ef6gx7efadnIlp}W`P0SabuBeCI1Wn*rHc?<iOhLCuAGjJ&K}gmpH{_m?`LbCJ*N2_Q#u3SC-4sfP@avZ'
    'pb^;@*H2k${OJZcJ5Q1*F;AP~=3Ct!9m(AS8WDNzp0xG3OICS0g;c7Q%hS`*Y}R?A=WM+eoxx6a&*8oyqO3B)vbsa0p75yil68hdkBEdik++3L'
    'R$BzoA3W3<kK&zZizt4uJ;q}EA{5<h0M-RRoK1nXc!8GS2Qq_<Zp9k>GujPXIZzJ<*5F;9piTZ2G7XkIP{Cwy8s<S@4aBqe5ZlxqU9&Ka68Z1~'
    '&CnSc4yABK<{zy$u#5R|mO9{fBPus#Zn9#T<N3?d2Ql&r=mja?Q43hl!_+8bDNIG%Yk)I$vn(;Vm9sRxX6JMegzou}#3mlN`n8{C5hZ|{w36B;'
    'Pcv=KwAcr|@^tfi7Np)jqY$-%HT_Ovt7BZL7j+U%Cmg!W=c+oUH*!0@KedAuDbsmArQ^1B9-ZsI>6CS7I?vt^Wc;R1t<?rdY#r7F3H&68HH?$q'
    'lDg>!Ssq;Z1E6i1nO2<bsg^u*XC4Ir<D>p0N|!72Mhd0ZY@IHWCB4hsNnsq8De0|=jozQbh(&5RBlLwAk1?E+doorQ-Itnsf}5}8J!fA&Uezq0'
    '>cdFiq^`jy@queg6jIej0J!-Ir3Vq8vZARlJ!zcI%T6_Tl~lr3z;p^lt59vW8&7eE)}79Aj5^NU28bs_8-#ppTPVEGr0S$rU^vhx=`QNkYziOT'
    '9ZiTxJDR31f@Kr<X9=HV(PEC`3I1un3Xt!rDS;wmOJR}G6q&F|Xs`97*RKw!F&O*w*W$q;9M+J^G#U<Vv+a1_K?-{%@I<aZj^mXZRp)Z*@Nzh4'
    'cvannk8v~SY}j>i^nE3i^2(G>y>VZ@XUx>i83?sk$o5l_9<(#5s*-g(1Db-GQ#TU$xK}p~-HQ}!4dF)|xC=%O#{ySGSW99HwD6dXI<i$!D()US'
    'MVWwSzD1x@(R>-{z{Bm%YzdOP^uDt3Zq;uJR9*PpC5t*=ljh#LYsXP+$dE)0@eWAl4%y0}H;^vMEq6Y<FRqD`MNX*%JkrDMfIe(CXu^h#IQprG'
    'n*HDaj;s9%S{EfX<|9Rix*Lu|>%<*N%~Q@O{uczN{7c(v2<!cfZ3+Zfocapu69#c>DCinf1>Mx2(Vg`1DiZk2Y~2@jIvsHyi4Pwd3bNLfpqs^h'
    'fR`+3?#Dfz^Pt6=<2JrQXio41Zt=R{ggklM=29?%5R_N9CQ;uV-SHp+5hF+|1_&Oj%Zn!^a2R<Em>;sQnGu$aNjJ^x*iBMqd_KGWU}KvN!cu@S'
    'Az_I;a0g4xho}$?1sG)`v}naIkY}R5e9xAq2F6_rZEP}YlL?Q$L}aUw|BrV5l`Zm&X}}Z>oGe&=jc0N%3-_y$Il|kYa^Z{0WbmLxOHKHsmppQX'
    '94C$vxGzeQSFFZDS?!SbZg6WL6U_D_oq+>dg>;%O<B0)!X!A{{CLVS~hgXrw*VEyQ<GlpF^kP+)*uKa8FMp(qT*_3cb_*^?Y%%%)Gmyl`$jjj?'
    'iff{BFnfrZ1BZ&>woj%rw5E{qqMR{FV;~PzLK$<xR8h*g)vqCjmpX_Y(i0n(>z|d!KiDVqj14h^v3E}h;Dt%R6H6+V28b=~70_$3?sPiZpi7wy'
    '#z+v8i=D#*`4yjX7M2?xf(xFRyB4fq`Z-p@4@>Bi@`L*lfiM#Am<$cPK$N2y!bSHfPiI@uBclxzo5d-aO%T3U)LOv?)KO$1UZ^5^(lrLY85wk9'
    'X~_PTqm(Y`9D^%^C{9U>V54Xk3@}&b(b2eWviR{Vz7^AaX~1cY#t`&&I;l8ecYXx%T$*`-KL=+5J`kS*`8#S~^LW~kHC}KJSf*#G>Y<*FrQx2k'
    'gt3WY(cElpIL;F8iQ||R+g7e|J&1cITm9nW*{|o0C<A@^a`EfM<=OAf<)`=OSGv{l_UgkKr5@GSxTbj35K$HO=d&yLes$e}>iQgBHDH=JEA<{R'
    '$T5%d&~p<A3iaM{5YnuPS68$c<%G7~2<sP8_!?pQ>qgb6`3VdEeuwb`It`PnveDGwhB>U^t=Z~M<6GKZqI4OB>cwl^ldU*o=`>K83KuBfqBB8-'
    '*;({qR&|fSC#vUZMG{6?fk?INYpemLlWmnLhMrO((Q)Gz(ISC*!LJ5oAdCt56fwc7V}HiHU_%x4f2?iwFK_=5EwTQcAu$N5Lq<J9GGJ+00IU`7'
    'czK&DOiCi$*rmX?x6Uiz3el^#I^V>Tz}?I}R|`+^uHX|;&JqOjFEotCGPLMjX;-;^@%Rv~8l;8^EAloHBrXzAmii9FE419kKtDB9cbqa7PPUkP'
    '57lA>?Lz@$Ut>`%fU5$RBV(~%-=S`gwHKN_(k<+*Pjm*(P3Iei-I67W$ZBNV?M*Pb?g-HSIYDZwG>n#AJ+W|N_LbqZr_jN|s#%kP(;+WKKD&D6'
    '_=RB~g-5NIAen2xoKh3Mk<2fJ>zJVCHgg^vY0DhNzwnfc0{oWK%OGPyhF{`-Tls5(i^-X_cqLRLf?lx@M*(T3F5?MB4UMGL#3DT=4vV&*sQ(mL'
    '@G>gIxZ)FhOIQg$N4-QHT7hz;BF&4*s2xJ=yD5n(5k>MHqzb(`h+efcqhP1iW+#8sJch*Hv=tH6VnnHj5{5ivPjrwnlne_+p#TC%YdhNQTTz_$'
    '@MSn81VOMZdTNDXrVZL~z2PvJny=Yc&tGB9%(V7KjC~P&l*Wv@aSorr=BBIczVe*A8m}L=rmqLf4_lk`)*+}1{CIqfxh(|z$K>6PxpzC}LC~=q'
    'AK#61b3fLzKdylv7uWxF9*duM(GMhZarw*nm-EYa=Z_{+3}YQg>E}N#-~ax2GJJHaM&n;A2Jx9El{C7?f;7Jv-?>oZh7X3aEUVRFv<%Y<S*g;A'
    'I7KF9Y3H%c$cD%;-Z0b>&3R$G45V5w#ZY*1!1So5ZAEi^tTxxNNi9nkN@--b1LR`}ZQ%Y_h7s)V6w#<=;rZAa21CqI5O@FOU-Oh9_P$|>$InN!'
    'in(iMqB1H`j8L?r=pCKtnT_b1tH7D{4r+&vP>j1qC^~1@dq*kSSM-io?445Y8L)WXoJ9w&XUtw4+Ee8^L<kO@*JxA!!!pc620a99SKrx^Q<a9%'
    ';C0bb?PT`e?y7O1z80jqu0(Z0V-L+1wYT6(4M#+xL(5?cd(d_(JFP{C!!+!We(|HGC7zV9N(BG6j(?$=G#JchYFo6F#=v>gUzASf06mqRhMDaE'
    'uVn`lz0qhmeAC0*3X-BnOTC0DE^IR#il=omh+u3UGUrOiek+cOjcT)A^rf<5)1{Jwrq;RKfyF!NwS$?Do{Fx@u0l`j1_K;t!G~I%6(3iXT>?3<'
    'T7O+Dl6HZKCVCa9*C>jhXkWoo$O3<dcQb(SFz^?C1_^S;LcAZ?@RXd>#0;z^R;N5>T{DQH@3m<D7th9Qj*H`6H6*(doL`Bdh6DVErIyWC)2H2T'
    'tem2IDX{d-7Bzqi?csrtQ=C>9pC<OUP1I~SF6uc>4Ol&X+tgEI3w(A2V@Kgq_a+eqX2@0Kj3WNNXb6#;ePZ@?0}X5ju0)6L7=V=@ctrOUz*gsz'
    'PE~5Pk;v1Md@;sbK))#8H9Jru-Vox&%dd+tsb&2O6|9JGk#qce%TL@M9l}sf2|1sop%fdFTBVwh%kSeb@O0liRmsH)UR;U8bn3Iw{wXkVL3FF_'
    'wg{yp#FRl180`Ip-hgRz?4tWA@}voh89DwIVlTMJaBzX{;=H9YcwU)!1OQlH+==Q;7^!73Uv}EjS-~cL*Pz<|A6z!n-X4ul)Ko;xPr3*15a6Vj'
    'PMtTWkRl>y@K*(mjIegZX$dkr@Gvc7e--5yMbOot=BFACuj!^*8;3PV1(!`zN`R&Nc`1_y;dtL_3_c(D(3Ey|b<L|zusskd^D5qo&+<ULkTR)J'
    ';?cLjpNn(3XwC3%@M&0E!8>K}CHDNy2|o0g(bLD5r2!ea@G3>U<_dHw^;NV9eeM5_{7fy!m7bh)m^hJY+&{_Un1rFC!|kb3jJqpZZ9G}B_&69A'
    '7i|2oY0CFp?WplSn}<J8OCir7Z)9oHS?Q5T9BPkRu@=05rx`^{q+W}oah68)5A9`p^Q6D4UtsI%>{l_U6}`6ZzUZ~%;kqU>>eGw8DpB@Fu`JnM'
    'hHt2eLX52Ftl#YG<j1s%+F&Dz-9soqqT=e3f_c$i(D`biw2(L_hT$Lo+Xki?+!GE>SXt}2D7-2(@U*S9f?8y|F^=RJ)yT3~$4RRdIFwFfY(&MR'
    'bZ2TZ5Zgz-;&m96znAS|^UiAJA|%Lg+2r()yl&-W#zbuC5aV!2L#Ck#wRX%Q216b`4-*#PFWS)N$|*@eB~KxYJ?>_(`yVsnV<r'
)
CURRENT_GALLERY_TEST_ORIGINAL_SHA256 = 'c14e6eb106629776ebf4007b45435677c59ccf37d004e56992365f977c0bc958'
CURRENT_GALLERY_TEST_NODE_SHA256 = {'_current_gallery_inverse': '803121764d9f36127d95050f93d8281c2b9c5fad306dcc165873d7e8fd2d616a', 'current_gallery_source_boundary': 'b50dd6c0146909387aa498924fcfeed47bbf336c900964afaeb6660993f2e5bd', 'current_gallery_test_boundary': 'ecba63be5896f222385c699503f5518128d8a8e047c42a43caaef40a2f663849', 'image_anchor_source_boundary': '568c49dd16361ba72276d2ca61e8e6efc24c136046d01f8bbededccadf03e41b', 'ContractTests.test_terminal_rejects_partial_false_and_nonfinite_cpu_proof': '80f582b1f5db36e5a0535752faec9d43c38ba0c58a7401ea2c471a14327b4fc0', 'SmoothAPTests.test_cpu_witness_requires_positive_nonnearest_loss_and_total_difference': 'e90f3b5bc49a3391aec6348678394b6facda6d4c122ab2d4ce956749686fb5d3', 'image_anchor_gradient_fixture': '9ac5089dd8b21fda88dd268f7c7347bc2a91cc622be4951a00ce2c9584adf488', 'ImageAnchorTests.test_both_arm_receipts_zero_and_target_difference_are_authenticated': 'd01ae35e197c5d28c12ed140af5e863f66bc917813722a01347bc5cd3b6133fc', 'ImageAnchorTests.test_previous_stdlib_cases_preserved_by_exact_required_inverses': '0557651d14ab59927b3070ccfa098367dbd0f9b35fbb8c028a0d8627f1438303', 'ImageAnchorTests.test_prospective_schemas_and_both_arm_ranking': '16d61a4dc2943d2129fd601c3dbd82915298172b02145f2640f6af98994a4caf', 'ImageAnchorTests.test_both_original_views_regress_to_canonical_image_with_common_e0': '1cf67304767ce907ea30a210b41cf89c0cc8cffd7aed056f439ef6130772f151', 'GalleryDual': 'cc7ac595232c679ac4e84d8c17680080919e10da52371b916f368b2492d8b52c', 'GalleryTensor': '3cdb638294bb218666d0b67c16b8af88180ee24182f5c987658ff476c444c8ab', 'CurrentGalleryTests': 'a05bed474f7079aa12096493d077787da85b2d4d39c5d30ddec49cf88d906cf1'}
CURRENT_GALLERY_TEST_BASE_AST_SHA256 = '081a64fc4fb79b0adaaefb93906d02c15104503c6e54c5594236f9154e239194'


def _current_gallery_inverse(tree, packed, packed_sha, pins, base_sha):
    """Restore exact listed sites; unmatched production/tests must equal the old AST."""
    import base64
    import zlib
    raw=zlib.decompress(base64.b85decode(packed))
    driver.require(hashlib.sha256(raw).hexdigest()==packed_sha,'current-gallery original nodes differ')
    originals=json.loads(raw)
    seen={}
    def name(node):
        return node.name if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else (
            node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else None)
    def restore(nodes,prefix=''):
        result=[]
        for node in nodes:
            key=prefix+name(node) if name(node) is not None else None
            if key in pins:
                driver.require(hashlib.sha256(ast.dump(node).encode()).hexdigest()==pins[key],
                               'exact current-gallery node differs: '+key)
                seen[key]=seen.get(key,0)+1
                if originals.get(key) is not None:result.append(ast.parse(originals[key]).body[0])
            else:
                if isinstance(node,ast.ClassDef):node.body=restore(node.body,node.name+'.')
                result.append(node)
        return result
    tree.body=restore(tree.body)
    driver.require(seen=={key:1 for key in pins},'exact current-gallery sites required')
    driver.require(hashlib.sha256(ast.dump(tree).encode()).hexdigest()==base_sha,
                   'current-gallery changed unrelated AST')
    return tree


def current_gallery_source_boundary(tree):
    tree = fullfeature_source_boundary(tree)
    return _current_gallery_inverse(tree,CURRENT_GALLERY_ORIGINAL_NODES,CURRENT_GALLERY_ORIGINAL_SHA256,
                                    CURRENT_GALLERY_NODE_SHA256,CURRENT_GALLERY_BASE_AST_SHA256)


def current_gallery_test_boundary(tree):
    tree = fullfeature_test_boundary(tree)
    # The pin constants themselves are data for the inverse, not inverse sites.
    names={'CURRENT_GALLERY_ORIGINAL_NODES','CURRENT_GALLERY_ORIGINAL_SHA256',
           'CURRENT_GALLERY_NODE_SHA256','CURRENT_GALLERY_BASE_AST_SHA256',
           'CURRENT_GALLERY_TEST_ORIGINAL_NODES','CURRENT_GALLERY_TEST_ORIGINAL_SHA256',
           'CURRENT_GALLERY_TEST_NODE_SHA256','CURRENT_GALLERY_TEST_BASE_AST_SHA256'}
    counts={name:0 for name in names}
    kept=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            counts[node.targets[0].id]+=1
        else:kept.append(node)
    driver.require(set(counts.values())=={1},'exact current-gallery test pin constants required')
    tree.body=kept
    return _current_gallery_inverse(tree,CURRENT_GALLERY_TEST_ORIGINAL_NODES,CURRENT_GALLERY_TEST_ORIGINAL_SHA256,
                                    CURRENT_GALLERY_TEST_NODE_SHA256,CURRENT_GALLERY_TEST_BASE_AST_SHA256)


def image_anchor_source_boundary(tree):
    """Exact prospective objective/witness inverse; every other AST node retained."""
    tree = current_gallery_source_boundary(tree)
    import base64
    import zlib
    raw = zlib.decompress(base64.b85decode(IMAGE_ANCHOR_ORIGINAL_NODES))
    driver.require(hashlib.sha256(raw).hexdigest() == IMAGE_ANCHOR_ORIGINAL_SHA256,
                   'image-anchor original source nodes differ')
    originals = {k: ast.parse(v).body[0] for k, v in json.loads(raw).items()}
    changed, result = {}, []
    for node in tree.body:
        name = (node.name if isinstance(node, ast.FunctionDef) else
                node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in IMAGE_ANCHOR_NODE_SHA256:
            driver.require(hashlib.sha256(ast.dump(node).encode()).hexdigest() == IMAGE_ANCHOR_NODE_SHA256[name],
                           'exact image-anchor reviewed node differs: ' + name)
            changed[name] = changed.get(name, 0) + 1
            if name in originals:
                result.append(copy.deepcopy(originals[name]))
        else:
            result.append(node)
    driver.require(changed == {k: 1 for k in IMAGE_ANCHOR_NODE_SHA256}, 'exact image-anchor sites required')
    tree.body = result
    driver.require(hashlib.sha256(ast.dump(tree).encode()).hexdigest() == IMAGE_ANCHOR_BASE_AST_SHA256,
                   'image-anchor changed unrelated production AST')
    return tree


def smooth_ap_source_boundary(tree):
    """Exact named-node inverse; reject any unreviewed objective/guard edit."""
    tree = image_anchor_source_boundary(tree)
    import base64
    import zlib
    raw = zlib.decompress(base64.b85decode(SMOOTH_AP_ORIGINAL_NODES))
    driver.require(hashlib.sha256(raw).hexdigest() == SMOOTH_AP_ORIGINAL_SHA256,
                   'SmoothAP original source nodes differ')
    originals = {k: ast.parse(v).body[0] for k, v in json.loads(raw).items()}
    changed, result = {}, []
    for node in tree.body:
        name = (node.name if isinstance(node, ast.FunctionDef) else
                node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in SMOOTH_AP_NODE_SHA256:
            driver.require(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() ==
                           SMOOTH_AP_NODE_SHA256[name], 'exact SmoothAP reviewed node differs: ' + name)
            changed[name] = changed.get(name, 0) + 1
            if name in originals:
                result.append(copy.deepcopy(originals[name]))
            elif name == 'json_sha256':
                result.append(copy.deepcopy(originals['mine']))
        else:
            result.append(node)
    driver.require(changed == {k: 1 for k in SMOOTH_AP_NODE_SHA256}, 'exact SmoothAP source sites required')
    tree.body = result
    return tree


def fresh_batch_source_boundary(tree):
    """Invert exactly the import/helper and two reviewed fresh-file sites."""
    tree = smooth_ap_source_boundary(tree)
    dump = lambda n: ast.dump(n, include_attributes=False)
    imported = ast.parse('from concurrent.futures import ThreadPoolExecutor').body[0]
    imports = [n for n in tree.body if dump(n) == dump(imported)]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batch_bound_files']
    driver.require(len(imports) == len(helpers) == 1, 'exact fresh batch import/helper required')
    tree.body = [n for n in tree.body if n not in imports + helpers]
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # Invert only the prospective non-CPU deadline; retain all historic AST pins.
    prospective_policy = ast.parse("def policy(phase):\n"
        "    require(phase in ('cpu', 'mechanics', 'train'), 'fixed phase required')\n"
        "    return {'seconds': 500 if phase == 'cpu' else 600, 'host_bytes': 8 * 1024**3,\n"
        "            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}\n").body[0]
    driver.require(dump(functions['policy']) == dump(prospective_policy), 'exact prospective runtime policy required')
    functions['policy'].body[1].value.values[0].orelse.value = 300
    changes = [
        ('admit_bundle',
         "batch_bound_files(guards, ((directory / name, digest) for name, digest in {**value['code'], **value['files']}.items()))\n"
         "for name, digest in {**value['code'], **value['files']}.items():\n"
         "    require((directory / name).stat().st_nlink == 1, 'bundle regular single-link ownership required')",
         "for name, digest in {**value['code'], **value['files']}.items():\n"
         "    bound_file(guards, directory / name, digest)\n"
         "    require((directory / name).stat().st_nlink == 1, 'bundle regular single-link ownership required')"),
        ('admit_bundle', "batch_bound_files(guards, env['files'].items())",
         "for path, digest in env['files'].items():\n    bound_file(guards, path, digest)"),
    ]
    for function, changed, original in changes:
        before, after = ast.parse(changed).body, ast.parse(original).body
        matches = []
        for node in ast.walk(functions[function]):
            for field, values in ast.iter_fields(node):
                if isinstance(values, list):
                    for i in range(len(values) - len(before) + 1):
                        if all(isinstance(a, ast.AST) and dump(a) == dump(b)
                               for a, b in zip(values[i:i + len(before)], before)):
                            matches.append((values, i))
        driver.require(len(matches) == 1, 'exact fresh batch site required: ' + function)
        values, i = matches[0]
        values[i:i + len(before)] = after
    driver.require(not any(isinstance(n, ast.Name) and n.id == 'batch_bound_files' for n in ast.walk(tree)),
                   'unexpected fresh batch site')
    expected = {'bound_file': '3db94d649ee69a5e3247924c59b7880d2fc4004242122e53717965b4beff7467',
                'admit_bundle': '22dadc6b2fc3933aad649e52cf208c33952c7a6c4a92dec8720c8f3499758b38',
                'load_inference': 'ba11c1f4fb8007e07121b861debfcf2453a49edaad096ac2d50311cf9c5f9609',
                'exit_rehash': 'd10e411cedd92f43f237a0271b8d0c193362e791ebd147c9910c2dc36c2206e8'}
    for name, digest in expected.items():
        driver.require(hashlib.sha256(dump(functions[name]).encode()).hexdigest() == digest,
                       'fresh batch retained function differs: ' + name)
    driver.require(hashlib.sha256(dump(tree).encode()).hexdigest() ==
                   '5a61ce29339d88826210452667d3b46650570a4bd9ba48d4d865c89647da346a',
                   'fresh batch changed retained module predicates')
    return tree


class FreshFileBatchTests(unittest.TestCase):
    def fixture(self, root, name):
        path = root / name
        path.write_bytes((name.encode() + b'\0') * 8192)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_serial_inventory_paths_errors_and_every_duplicate_read(self):
        self.assertTrue(hasattr(driver, 'batch_bound_files'), 'missing batch_bound_files')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self.fixture(root, 'first'), self.fixture(root, 'second')
            items = [first, second, first]
            serial = {'existing': 'authority'}
            paths = [driver.bound_file(serial, *item) for item in items]
            guards, reads = {'existing': 'authority'}, []
            real = driver.bound_file
            def observe(private, path, digest):
                self.assertEqual(private, {})
                self.assertIsNot(private, guards)
                result = real(private, path, digest)
                reads.append(str(result))
                return result
            with patch.object(driver, 'bound_file', observe):
                self.assertEqual(driver.batch_bound_files(guards, iter(items)), paths)
            self.assertEqual(list(guards.items()), list(serial.items()))
            self.assertCountEqual(reads, [str(p) for p, _ in items])
            alias = root / 'alias'; alias.symlink_to(first[0])
            fifo = root / 'fifo'; os.mkfifo(fifo)
            cases = [([(alias, first[1])], {}), ([(fifo, first[1])], {}),
                     ([(root / 'missing', first[1])], {}), ([(Path('relative'), first[1])], {}),
                     ([(first[0], 'BAD')], {}), ([(first[0], '0' * 64)], {}),
                     ([first, (first[0], second[1])], {}),
                     ([second, first], {str(first[0]): '0' * 64})]
            for entries, initial in cases:
                def serial_check():
                    values = dict(initial)
                    return [real(values, *item) for item in entries]
                with self.subTest(entries=entries):
                    with self.assertRaises(ValueError) as old:
                        serial_check()
                    values = dict(initial)
                    with self.assertRaises(ValueError) as new:
                        driver.batch_bound_files(values, entries)
                    self.assertEqual(str(new.exception), str(old.exception))
                    self.assertEqual(values, initial)
            self.assertEqual(driver.batch_bound_files(guards, []), [])
            # Both conflicting authorities can individually match fresh bytes:
            # the owner must still reject without publishing either result.
            saved = first[0].read_bytes()
            changed = b'x' + saved[1:]
            next_digest = hashlib.sha256(changed).hexdigest()
            first_done = threading.Event()
            def mutate_duplicate(private, path, digest):
                if digest == next_digest:
                    self.assertTrue(first_done.wait(5))
                    path.write_bytes(changed)
                    return real(private, path, digest)
                result = real(private, path, digest)
                first_done.set()
                return result
            values = {'existing': 'authority'}
            try:
                with patch.object(driver, 'bound_file', mutate_duplicate), self.assertRaisesRegex(
                        ValueError, 'conflicting FILE authority'):
                    driver.batch_bound_files(values, [first, (first[0], next_digest)])
                self.assertEqual(values, {'existing': 'authority'})
            finally:
                first[0].write_bytes(saved)

    def test_fresh_same_size_restored_mtime_and_consumed_page_advice(self):
        self.assertTrue(hasattr(driver, 'batch_bound_files'), 'missing batch_bound_files')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'large'
            raw = b'a' * (2 * 1024**2 + 31)
            path.write_bytes(raw)
            item = (path, hashlib.sha256(raw).hexdigest())
            guards, advice = {}, []
            real_advice = os.posix_fadvise
            def observe(fd, offset, count, flag):
                advice.append((offset, count, flag))
                return real_advice(fd, offset, count, flag)
            with patch.object(driver.os, 'posix_fadvise', observe):
                for _ in range(2):
                    self.assertEqual(driver.batch_bound_files(guards, [item]), [path])
            self.assertEqual(advice, [(0, 1024**2, os.POSIX_FADV_DONTNEED),
                (1024**2, 1024**2, os.POSIX_FADV_DONTNEED),
                (2 * 1024**2, 31, os.POSIX_FADV_DONTNEED)] * 2)
            stamp = path.stat()
            path.write_bytes(b'b' + raw[1:])
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(path.stat().st_size, len(raw))
            self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'current FILE bytes differ'):
                    driver.batch_bound_files(guards, [item])
            self.assertEqual(guards, {str(path): item[1]})

    def test_four_workers_join_failure_no_publication_and_owner_order(self):
        self.assertTrue(hasattr(driver, 'batch_bound_files'), 'missing batch_bound_files')
        owner = threading.get_ident()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            items = [self.fixture(root, str(i)) for i in range(12)]
            real = driver.bound_file
            for failing in (False, True):
                barrier, lock = threading.Barrier(4, timeout=5), threading.Lock()
                active = peak = started = 0
                finished, workers = [], set()
                sentinel = ValueError('injected worker failure')
                updates = []
                class Guards(dict):
                    def update(self, values):
                        updates.append((threading.get_ident(), list(values.items())))
                        super().update(values)
                guards = Guards(existing='authority')
                def observe(private, path, digest):
                    nonlocal active, peak, started
                    with lock:
                        started += 1
                        slot = started
                        active += 1
                        peak = max(peak, active)
                        workers.add(threading.current_thread())
                    try:
                        self.assertIsNot(private, guards)
                        self.assertEqual(private, {})
                        if slot <= 4: barrier.wait()
                        self.assertEqual(guards, {'existing': 'authority'})
                        result = real(private, path, digest)
                        if failing and path == items[0][0]: raise sentinel
                        return result
                    finally:
                        with lock:
                            active -= 1
                            finished.append(path)
                with patch.object(driver, 'bound_file', observe):
                    if failing:
                        with self.assertRaises(ValueError) as caught:
                            driver.batch_bound_files(guards, items)
                        self.assertIs(caught.exception, sentinel)
                        self.assertEqual(guards, {'existing': 'authority'})
                        self.assertEqual(updates, [])
                    else:
                        self.assertEqual(driver.batch_bound_files(guards, items), [p for p, _ in items])
                        expected = [('existing', 'authority'), *[(str(p), h) for p, h in items]]
                        self.assertEqual(list(guards.items()), expected)
                        self.assertEqual(updates, [(owner, expected)])
                self.assertEqual(peak, 4)
                self.assertEqual(active, 0)
                self.assertCountEqual(finished, [p for p, _ in items])
                self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_exact_two_sites_original_bytes_and_predicate_correspondence(self):
        self.assertTrue(hasattr(driver, 'batch_bound_files'), 'missing batch_bound_files')
        source = PATH.read_text()
        tree = ast.parse(source)
        bound = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bound_file')
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source, bound).encode()).hexdigest(),
                         '193c1b2f76f5b8d5c9e5486cc66514a234b3fda7acfa43dee4aa413de57af5f7')
        exit_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source, exit_node).encode()).hexdigest(),
                         'ab782b58b66388b7cee066836931e3ef9a4f1713729a1dadad33e8b87fa28a99')
        fresh_batch_source_boundary(copy.deepcopy(tree))
        for name, statement in (
                ('admit_bundle', "batch_bound_files(guards, env['files'].items())"),):
            mutant = copy.deepcopy(tree)
            target = ast.dump(ast.parse(statement).body[0])
            class Omit(ast.NodeTransformer):
                def visit_Expr(self, node):
                    return ast.Pass() if ast.dump(node) == target else self.generic_visit(node)
            with self.subTest(omitted=name), self.assertRaises(ValueError):
                fresh_batch_source_boundary(Omit().visit(mutant))
        mutant = ast.parse(source.replace('exit_reader.bound_file({}, p, h)', 'bound_file({}, p, h)'))
        with self.subTest(mutant='replace_fresh_exit_reader'), self.assertRaises(ValueError):
            fresh_batch_source_boundary(mutant)
        for fragment in ("== 1, 'bundle regular single-link", "env['native_files'].get(p) == h"):
            mutant = ast.parse(source.replace(fragment, fragment.replace('== 1', '>= 1').replace('== h', '!= h')))
            with self.subTest(predicate=fragment), self.assertRaises(ValueError):
                fresh_batch_source_boundary(mutant)


def completion_source_boundary(tree):
    """Invert only the reviewed completion edits, with exact per-site counts."""
    tree = fresh_batch_source_boundary(tree)
    dump = lambda n: ast.dump(n, include_attributes=False)
    added = {'exit_admission_adapter', 'authenticate_bundle_environment'}
    for name in added:
        driver.require(sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) == 1,
                       'exact completion definition required: ' + name)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in added)]
    removals = {
        ('exit_rehash', "exit_reader = context['legacy']['original'].FlatAdmission()"): 1,
        ('exit_rehash', 'del exit_reader'): 1,
        ('run', "post_run_reader = legacy['original'].FlatAdmission()"): 1,
        ('run', 'del post_run_reader'): 1,
        ('qualify_bundle', "authenticate_bundle_environment(context, bundle['environment'])"): 1,
    }
    replacements = {
        ('exit_rehash', "api.audit_origins(context['legacy'], admission=exit_reader, require_exact=context['args'].phase != 'cpu')"):
            ("api.audit_origins(context['legacy'], admission=context['legacy']['original'].FlatAdmission(), require_exact=context['args'].phase != 'cpu')", 1),
        ('exit_rehash', "exit_admission_adapter(context, api, exit_reader)(context['fit_context'])"):
            ("api.exit_rehash(context['fit_context'])", 1),
        ('exit_rehash', 'exit_reader.bound_file({}, p, h)'): ('bound_file({}, p, h)', 1),
        ('run', "api.audit_origins(legacy, admission=post_run_reader, require_exact=args.phase != 'cpu')"):
            ("api.audit_origins(legacy, admission=legacy['original'].FlatAdmission(), require_exact=args.phase != 'cpu')", 1),
        ('run', "post_run_reader.bound_file(context['guards'], p, h)"):
            ("bound_file(context['guards'], p, h)", 1),
        ('qualify_bundle', "bundle = portable.read_json({'path': str(directory / 'bundle.json'), 'sha256': sha}, {})"):
            ('bundle, _ = portable.admit_bundle(directory, sha)', 1),
    }
    removed, replaced = {}, {}

    class Inverse(ast.NodeTransformer):
        function = None

        def visit_FunctionDef(self, node):
            prior, self.function = self.function, node.name
            result = self.generic_visit(node)
            self.function = prior
            return result

        def visit(self, node):
            if isinstance(node, ast.stmt):
                for key in removals:
                    if self.function == key[0] and dump(node) == dump(ast.parse(key[1]).body[0]):
                        removed[key] = removed.get(key, 0) + 1
                        return None
            for key, (original, _) in replacements.items():
                changed = ast.parse(key[1]).body[0]
                if isinstance(changed, ast.Expr):
                    changed = changed.value
                    original = ast.parse(original, mode='eval').body
                else:
                    original = ast.parse(original).body[0]
                if self.function == key[0] and dump(node) == dump(changed):
                    replaced[key] = replaced.get(key, 0) + 1
                    return ast.copy_location(original, node)
            return super().visit(node)

    tree = Inverse().visit(tree)
    driver.require(removed == removals and replaced == {k: count for k, (_, count) in replacements.items()},
                   'exact completion scheduling sites required')
    driver.require(hashlib.sha256(dump(tree).encode()).hexdigest() ==
                   '08bdc0910571d5638f9e49ba9b6fae585d47374ee1759020151b9cdd704ec2d2',
                   'completion changed retained predicates')
    return tree


class FreshOriginAuditTests(unittest.TestCase):
    """Genuine private audit, reader and collector over stdlib origin fixtures."""
    @classmethod
    def setUpClass(cls):
        path = PATH.with_name('test_siglip2_nearest_ranking.py')
        spec = importlib.util.spec_from_file_location('compact_origin_fixtures', path)
        cls.fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixtures)

    @contextmanager
    def composition(self):
        with TemporaryDirectory() as directory, patch.dict(sys.modules):
            f = self.fixtures.NativeAdmissionFixture(Path(directory))
            # Bind the genuine collector before the real API freezes dependencies.
            f.source = f.module('qualify_siglip2_substrate_cpu.py')
            f.legacy['source_driver'] = f.source
            f.context['guards'][f.source.__file__] = hashlib.sha256(Path(f.source.__file__).read_bytes()).hexdigest()
            f.legacy['prior']['source_driver'] = f.source
            modules, cpu, warm = {}, {'files': {}, 'modules': {}}, {'files': {}, 'modules': {}}
            for name, proof in (('cpu', cpu), ('warm', warm)):
                path = Path(directory) / (name + '.py')
                path.write_bytes(b'# original origin fixture\n')
                alias = 'compact_origin_fixture.' + name
                spec = importlib.util.spec_from_file_location(alias, path)
                modules[alias] = importlib.util.module_from_spec(spec)
                proof['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                proof['modules'][alias] = str(path)
            f.legacy['selected']['packages'] = {'compact_origin_fixture': {'root': directory}}
            f.legacy['selected']['source_cpu']['origins'] = cpu
            f.legacy['warm_record']['origins'] = warm
            f.cpu, f.warm, f.loaded = cpu, warm, modules
            f.observed = {**cpu['files'], **warm['files'], **f.files}
            f.mapped = list(f.files)
            f.readers, f.reads = [], []
            f.original_state = [(m, dict(vars(m))) for m in (f.old, f.fitter, f.original, f.source, f.extract)]
            read_text, open_file = Path.read_text, Path.open

            def maps(path, *args, **kwargs):
                if str(path) == '/proc/self/maps':
                    return '\n'.join('0-1 r--p 0 00:00 0 ' + p for p in f.mapped)
                return read_text(path, *args, **kwargs)

            def observed_open(path, *args, **kwargs):
                if str(path) in f.observed:
                    code = sys._getframe(1).f_code
                    if code is f.extract.sha.__code__:
                        f.reads.append(('origin_sha', str(path)))
                    elif code is f.old.bound_file.__code__:
                        f.reads.append(('duplicate_sha', str(path)))
                return open_file(path, *args, **kwargs)

            def acquire(context):
                api = self.fixtures.driver.native_source_api(context)

                def audit(legacy, *args, **kwargs):
                    f.readers.append(kwargs.get('admission'))
                    return api.audit_origins(legacy, *args, **kwargs)

                return SimpleNamespace(audit_origins=audit)

            f.context.update(nearest=SimpleNamespace(native_source_api=acquire), args=SimpleNamespace(phase='mechanics'))
            with patch.dict(sys.modules, modules), patch.object(Path, 'read_text', maps), patch.object(Path, 'open', observed_open):
                f.api = f.admit()
                yield f
                f.unchanged_originals(self)
                self.assertFalse(any(n.split('.')[0] in driver.NATIVE for n in sys.modules))

    def boundaries(self, f):
        # Execute each actual compact audit expression with the real private API.
        # Whole-module correspondence below protects all surrounding operations.
        result = []
        for fn in ast.parse(PATH.read_text()).body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call) and
                            isinstance(n.func, ast.Attribute) and n.func.attr == 'audit_origins'),
                           key=lambda n: n.lineno)
            for call in calls:
                code = compile(ast.Expression(body=call), str(PATH), 'eval')

                def invoke(code=code):
                    api = f.context['nearest'].native_source_api(f.context)
                    reader = f.original.FlatAdmission()
                    return eval(code, vars(driver), {'context': f.context, 'legacy': f.legacy,
                                'args': f.context['args'], 'api': api,
                                'exit_reader': reader, 'post_run_reader': reader})

                result.append((fn.name, invoke))
        self.assertEqual([name for name, _ in result], ['prepare_native', 'gpu_run', 'exit_rehash', 'exit_rehash', 'run'])
        return result

    def test_fresh_original_reader_and_one_genuine_origin_hash_per_boundary(self):
        with self.composition() as f:
            legacy_guards, prior_guards = dict(f.legacy['guards']), dict(f.legacy['prior']['guards'])
            f.api.audit_origins(f.legacy)
            self.assertCountEqual(f.reads, [(kind, p) for p in f.observed
                                           for kind in ('origin_sha', 'duplicate_sha')])
            f.admission.verified.update(f.observed)  # Startup cache cannot qualify a later audit.
            for phase in ('cpu', 'mechanics', 'train'):
                f.context['args'].phase = phase
                for name, invoke in self.boundaries(f):
                    with self.subTest(phase=phase, boundary=name):
                        f.reads.clear()
                        invoke()
                        reader = f.readers[-1]
                        self.assertIs(type(reader), f.original.FlatAdmission)
                        self.assertIsNot(reader, f.admission)
                        self.assertTrue(all(reader is not old for old in f.readers[:-1]))
                        self.assertEqual(reader.verified, set(f.observed))
                        self.assertEqual(reader.entries, {p: (h, Path(p).stat().st_size) for p, h in f.observed.items()})
                        self.assertEqual(reader.json_bytes, {})
                        self.assertCountEqual(f.reads, [('origin_sha', p) for p in f.observed])
                        self.assertEqual(f.legacy['guards'], {**legacy_guards, **f.observed})
                        self.assertEqual(f.legacy['prior']['guards'], {**prior_guards, **f.observed})
                        self.assertEqual(f.legacy['origins']['files'], f.observed)
            self.assertEqual(len(f.readers), 15)

    def test_private_audit_predicates_and_guard_correspondence(self):
        with self.composition() as f:
            unknown = Path(f.root) / 'unknown.so'
            unknown.write_bytes(b'unknown')
            supplemental = next(iter(f.files))
            cases = [
                ('accepted', {'require_exact': True}, None),
                ('initial', {'initial': True}, 'original CPU'),
                ('initial_accepted', {'initial': True}, None),
                ('file_conflict', {}, 'conflicting original'),
                ('module_conflict', {}, 'conflicting original'),
                ('unknown_file', {}, 'unknown or changed'),
                ('unknown_module', {}, 'unknown or changed'),
                ('outside_package', {}, 'loaded native module origin differs'),
                ('legacy_guard', {}, 'conflicting'),
                ('supplement_guard', {}, 'conflicting'),
                ('prior_guard', {}, 'original native origin changed'),
                ('missing_one', {'require_exact': True}, 'exact four'),
                ('cpu_subset', {'require_exact': False}, None),
                ('missing_native', {}, 'missing native'),
            ]
            original_guards = dict(f.legacy['guards'])
            for name, kwargs, error in cases:
                outcomes = []
                for admitted in (False, True):
                    f.legacy['guards'] = dict(original_guards)
                    f.legacy['prior']['guards'] = {}
                    f.legacy['selected']['source_cpu']['origins'] = copy.deepcopy(f.cpu)
                    f.legacy['warm_record']['origins'] = copy.deepcopy(f.warm)
                    f.legacy.pop('origins', None)
                    f.mapped = list(f.files)
                    loaded = dict(f.loaded)
                    if name == 'initial_accepted':
                        f.legacy['selected']['source_cpu']['origins'] = {
                            'files': dict(f.observed), 'modules': {**f.cpu['modules'], **f.warm['modules']}}
                    elif name == 'file_conflict':
                        f.legacy['warm_record']['origins']['files'].update({p: '0' * 64 for p in f.cpu['files']})
                    elif name == 'module_conflict':
                        f.legacy['warm_record']['origins']['modules'].update({n: '/conflict.py' for n in f.cpu['modules']})
                    elif name == 'unknown_file':
                        f.mapped.append(str(unknown))
                    elif name == 'unknown_module':
                        loaded['compact_origin_fixture.unknown'] = next(iter(f.loaded.values()))
                    elif name == 'outside_package':
                        loaded['compact_origin_fixture.unknown'] = SimpleNamespace(__file__=f.source.__file__)
                    elif name == 'legacy_guard':
                        f.legacy['guards'][next(iter(f.cpu['files']))] = '0' * 64
                    elif name == 'supplement_guard':
                        f.legacy['guards'][supplemental] = '0' * 64
                    elif name == 'prior_guard':
                        f.legacy['prior']['guards'][next(iter(f.cpu['files']))] = '0' * 64
                    elif name in ('missing_one', 'cpu_subset', 'missing_native'):
                        f.mapped.remove(supplemental)
                        if name == 'missing_native':
                            alias = 'compact_origin_fixture.supplemental'
                            loaded[alias] = SimpleNamespace(__file__=supplemental)
                            f.legacy['warm_record']['origins']['modules'][alias] = supplemental
                    reader = f.original.FlatAdmission() if admitted else None
                    with self.subTest(case=name, admitted=admitted), patch.dict(sys.modules, loaded):
                        if error is None:
                            f.api.audit_origins(f.legacy, admission=reader, **kwargs)
                        else:
                            with self.assertRaisesRegex(ValueError, error):
                                f.api.audit_origins(f.legacy, admission=reader, **kwargs)
                        outcomes.append((dict(f.legacy['guards']), dict(f.legacy['prior']['guards']),
                                         copy.deepcopy(f.legacy.get('origins'))))
                self.assertEqual(outcomes[0], outcomes[1], name)

    def test_current_bytes_tamper_and_collector_failures_propagate(self):
        with self.composition() as f:
            path = Path(next(iter(f.cpu['files'])))
            saved, stamp = path.read_bytes(), path.stat()
            sentinel = ValueError('origin read failed')
            real_open = Path.open
            for name, invoke in self.boundaries(f):
                with self.subTest(boundary=name):
                    invoke()
                    try:
                        path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        self.assertEqual(path.stat().st_size, len(saved))
                        self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns)
                        with self.assertRaisesRegex(ValueError, 'unknown or changed'):
                            invoke()
                    finally:
                        path.write_bytes(saved)

                    def failed_open(value, *args, **kwargs):
                        if value == path and sys._getframe(1).f_code is f.extract.sha.__code__:
                            raise sentinel
                        return real_open(value, *args, **kwargs)

                    with patch.object(Path, 'open', failed_open), self.assertRaises(ValueError) as caught:
                        invoke()
                    self.assertIs(caught.exception, sentinel)

                    def failed_dependency(value, *args, **kwargs):
                        if str(value) == f.source.__file__:
                            raise sentinel
                        return real_open(value, *args, **kwargs)

                    with patch.object(Path, 'open', failed_dependency), self.assertRaises(ValueError) as caught:
                        invoke()
                    self.assertIs(caught.exception, sentinel)
                    invoke()


    def test_nested_exit_reader_composition_and_fail_closed_mutations(self):
        self.assertTrue(hasattr(driver, 'exit_admission_adapter'), 'missing reviewed exit reader adapter')
        with self.composition() as f:
            f.context['nearest'] = SimpleNamespace(native_source_api=self.fixtures.driver.native_source_api,
                                                   require_no_model=self.fixtures.driver.require_no_model)
            f.context.update(started=driver.time.perf_counter(), phase_seconds={})
            prior = f.legacy['selected']['genuine']['prior']
            # The real CPU collector also owns the FIT validator and bootstrap.
            # Supply their complete stdlib inputs instead of replacing predicates.
            image_root = Path(prior['fit']['dataset_root'])
            image_directory = image_root / 'Img/img'; image_directory.mkdir(parents=True)
            paths = [p.rename(image_directory / p.name) for p in prior['all_images']]
            classes = ['class-' + str(i) for i in range(2004)]
            prior['fit'].update(schema='native256-frozen-fit-manifest-v1', fit_images=13283,
                fit_identities=2004, held_images_read=0, quality_read=False,
                source_features_reused=False, teacher_state_reused=False,
                targets=[i % 2004 for i in range(13283)], class_names=classes,
                rows=[{'relative_path': 'Img/img/' + p.name, 'train_row': i, 'product': classes[i % 2004],
                       'image_sha256': hashlib.sha256(b'').hexdigest()} for i, p in enumerate(paths)])
            prior['all_images'] = paths
            prior['images'] = f.source.fit_rows(f.extract, prior['fit'])
            source_root = f.root / 'source'; source_root.mkdir()
            for name in f.source.FILES: shutil.copyfile(PATH.with_name(name), source_root / name)
            source_code = {n: hashlib.sha256((source_root / n).read_bytes()).hexdigest() for n in f.source.FILES}
            source_execution = f.write_json('source/execution.json', source_code)
            prior.update(root=source_root, code=source_code,
                         args=SimpleNamespace(execution_sha256=source_execution['sha256']))
            name = 'extract_siglip2_vision_source'
            spec = importlib.util.spec_from_file_location(name, source_root / (name + '.py'))
            extract = importlib.util.module_from_spec(spec); spec.loader.exec_module(extract)
            sys.modules[name] = extract
            # One non-origin path belongs to each of the four stage inventories.
            shared = f.bulk[0]
            for guards in (prior['guards'], f.legacy['selected']['genuine']['guards'],
                           f.legacy['guards'], f.context['fit_context']['guards'], f.context['guards']):
                guards[shared['path']] = shared['sha256']
            inventories = [prior['guards'], f.legacy['selected']['genuine']['guards'],
                           f.legacy['guards'], f.context['fit_context']['guards'], f.context['guards']]
            reader = f.original.FlatAdmission()
            f.api.audit_origins(f.legacy, admission=reader, require_exact=True)
            initial_guards = [dict(g) for g in inventories]
            admitted = driver.exit_admission_adapter(f.context, f.api, reader)
            private_fit = next(c.cell_contents for c in admitted.__closure__ or ()
                               if isinstance(c.cell_contents, FunctionType) and c.cell_contents.__name__ == 'exit_rehash')
            private_quad = private_fit.__globals__['_compact_quadratic_exit']
            self.assertIs(private_quad.__globals__['audit_origins'], f.api.audit_origins)
            self.assertIs(private_quad.__globals__['_compact_exit_reader'], reader)
            self.assertIs(private_fit.__globals__['_compact_exit_reader'], reader)
            for module, private, substitutions in (
                    (f.old, private_quad, [("context['original'].FlatAdmission()", '_compact_exit_reader')]),
                    (f.fitter, private_fit, [("context['old'].exit_rehash(context['legacy'])", "_compact_quadratic_exit(context['legacy'])"),
                                             ('bound_file({}, path, digest)', '_compact_exit_reader.bound_file({}, path, digest)')])):
                self.assertIsNot(private.__globals__, vars(module))
                node = next(n for n in ast.parse(Path(module.__file__).read_bytes()).body
                            if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
                original = copy.deepcopy(node)
                dump = lambda n: ast.dump(n, include_attributes=False)
                for before, after in substitutions:
                    class Change(ast.NodeTransformer):
                        count = 0
                        def visit(self, n):
                            if dump(n) == dump(ast.parse(before, mode='eval').body):
                                self.count += 1
                                return ast.copy_location(ast.parse(after, mode='eval').body, n)
                            return super().visit(n)
                    change = Change()
                    node = change.visit(node)
                    self.assertEqual(change.count, 1)
                expected = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), module.__file__, 'exec')
                self.assertEqual(private.__code__, next(c for c in expected.co_consts if getattr(c, 'co_name', None) == 'exit_rehash'))
                for before, after in reversed(substitutions):
                    before, after = after, before
                    change = Change()
                    node = change.visit(node)
                    self.assertEqual(change.count, 1)
                self.assertEqual(dump(node), dump(original))

            with redirect_stdout(io.StringIO()):
                f.api.exit_rehash(f.context['fit_context'])
            expected_guards = [dict(g) for g in inventories]
            for guards, saved in zip(inventories, initial_guards):
                guards.clear(); guards.update(saved)
            f.context['fit_context']['phase_seconds'].clear()
            f.reads.clear()
            reads, real_open = [], Path.open
            def count_open(path, *args, **kwargs):
                if str(path) in f.observed and sys._getframe(1).f_code is f.extract.sha.__code__:
                    f.reads.append(('origin_sha', str(path)))
                if str(path) == shared['path']:
                    reads.append(sys._getframe(1).f_code)
                return real_open(path, *args, **kwargs)
            with patch.object(Path, 'open', count_open), redirect_stdout(io.StringIO()):
                admitted(f.context['fit_context'])
                for path, digest in f.context['guards'].items():
                    reader.bound_file({}, path, digest)
            self.assertEqual(reads, [f.original.bound_file.__code__])
            self.assertEqual([dict(g) for g in inventories], expected_guards)
            self.assertEqual(reader.entries[shared['path']], (shared['sha256'], Path(shared['path']).stat().st_size))
            # The nested collector still physically rereads each origin.
            self.assertCountEqual([r for r in f.reads if r[0] == 'origin_sha'], [('origin_sha', p) for p in f.observed])
            with self.assertRaises(ValueError): reader.bound_file({}, shared['path'], '0' * 64)
            with self.assertRaises(ValueError): reader.bound_file({}, shared['path'], shared['sha256'], size=0)
            with self.assertRaises(ValueError): reader.bound_file({shared['path']: '0' * 64}, shared['path'], shared['sha256'])
            alias = f.root / 'alias'; alias.symlink_to(shared['path'])
            with self.assertRaises(ValueError): reader.bound_file({}, alias, shared['sha256'])

            # No garbage collection may rescue a successful boundary's lifetime.
            reader_ref = weakref.ref(reader)
            collecting = driver.gc.isenabled(); driver.gc.disable()
            try:
                del reader, admitted, private_fit, private_quad, private
                self.assertIsNone(reader_ref(), 'successful private exit retained its reader')
            finally:
                if collecting: driver.gc.enable()

            def candidate():
                reader = f.original.FlatAdmission()
                f.api.audit_origins(f.legacy, admission=reader, require_exact=True)
                admitted = driver.exit_admission_adapter(f.context, f.api, reader)
                fit = next(c.cell_contents for c in admitted.__closure__ or ()
                           if isinstance(c.cell_contents, FunctionType) and c.cell_contents.__name__ == 'exit_rehash')
                return reader, admitted, fit, fit.__globals__['_compact_quadratic_exit']

            for owner, name in (('fit', '_compact_exit_reader'), ('fit', '_compact_quadratic_exit'), ('quad', 'audit_origins')):
                reader, admitted, private_fit, private_quad = candidate()
                values = (private_fit if owner == 'fit' else private_quad).__globals__
                replacement = f.original.FlatAdmission() if name == '_compact_exit_reader' else lambda *a, **kw: None
                with self.subTest(global_name=name), patch.dict(values, {name: replacement}), self.assertRaises(ValueError):
                    admitted(f.context['fit_context'])
                values.clear()  # patch.dict restores its snapshot after rejection.
            for name in ('bound_file', 'register', 'all_fit_images'):
                reader, admitted, private_fit, private_quad = candidate()
                with self.subTest(shadow=name), patch.object(reader, name, lambda *a: None), self.assertRaises(ValueError):
                    admitted(f.context['fit_context'])
            for owner in ('fit', 'quad'):
                for name, replacement in (('__code__', (lambda value: None).__code__),
                                           ('__defaults__', (None,)), ('__kwdefaults__', {'unexpected': True})):
                    reader, admitted, private_fit, private_quad = candidate()
                    fn = private_fit if owner == 'fit' else private_quad
                    saved = getattr(fn, name)
                    try:
                        setattr(fn, name, replacement)
                        with self.subTest(function=owner, attribute=name), self.assertRaises(ValueError):
                            admitted(f.context['fit_context'])
                    finally: setattr(fn, name, saved)
            reader, admitted, private_fit, private_quad = candidate()
            cells = dict(zip(admitted.__code__.co_freevars, admitted.__closure__))
            cell = cells['authenticate']; original_auth = cell.cell_contents
            try:
                cell.cell_contents = lambda: None
                with self.assertRaises(ValueError): admitted(f.context['fit_context'])
            finally: cell.cell_contents = original_auth
            reader, admitted, private_fit, private_quad = candidate()
            with patch.dict(f.fitter.ORIGINAL_CODE, {'unexpected': '0' * 64}), self.assertRaises(ValueError):
                admitted(f.context['fit_context'])
            unknown = f.root / 'unknown.so'; unknown.write_bytes(b'unknown')
            reader, admitted, private_fit, private_quad = candidate()
            f.mapped.append(str(unknown))
            f.context['fit_context']['phase_seconds'].clear()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'unknown or changed'):
                admitted(f.context['fit_context'])
            f.mapped.remove(str(unknown))
            supplemental = next(iter(f.files)); f.mapped.remove(supplemental)
            with self.assertRaisesRegex(ValueError, 'exact four'):
                f.api.audit_origins(f.legacy, admission=f.original.FlatAdmission(), require_exact=True)
            f.mapped.append(supplemental)
            # The same stage predicates must fail before any union rescue.
            for name, value in (('images', ['changed FIT']), ('all_images', [])):
                reader, admitted, private_fit, private_quad = candidate()
                with patch.dict(prior, {name: value}), redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'FIT image resolution'):
                    f.context['fit_context']['phase_seconds'].clear()
                    admitted(f.context['fit_context'])
            f.context['fit_context']['phase_seconds'].clear()
            with redirect_stdout(io.StringIO()):
                f.api.exit_rehash(f.context['fit_context'])
            self.assertEqual([dict(g) for g in inventories], expected_guards)
            # A fresh exit cannot reuse startup bytes, even with restored mtime.
            path = Path(shared['path']); saved, stamp = path.read_bytes(), path.stat()
            try:
                path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:]); os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                fresh = f.original.FlatAdmission()
                f.api.audit_origins(f.legacy, admission=fresh, require_exact=True)
                f.context['fit_context']['phase_seconds'].clear()
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    driver.exit_admission_adapter(f.context, f.api, fresh)(f.context['fit_context'])
            finally: path.write_bytes(saved)
            # Exercise the actual compact exit as well: its last independent
            # boundary must start after disposal and still hash current origins.
            def make_closure(name, members):
                root = f.root / name; root.mkdir()
                for member in members: shutil.copyfile(PATH.with_name(member), root / member)
                code = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in members}
                execution = f.write_json(name + '/execution.json', code)
                return {'root': str(root), 'code': code, 'execution_sha256': execution['sha256']}
            own = make_closure('compact', driver.FILES)
            nearest = make_closure('nearest', driver.NEAREST['code'])
            f.context.update(root=Path(own['root']), code=own['code'],
                args=SimpleNamespace(phase='mechanics', execution_sha256=own['execution_sha256']))
            adapter, timer = driver.exit_admission_adapter, driver.timed
            native = Path(supplemental); saved, stamp = native.read_bytes(), native.stat()
            for tamper in (False, True):
                reader_refs = []
                def track(context, api, reader):
                    reader_refs.append(weakref.ref(reader))
                    return adapter(context, api, reader)
                @contextmanager
                def phase(context, name):
                    if name == 'post_exit_api_authentication':
                        self.assertEqual(len(reader_refs), 1)
                        self.assertIsNone(reader_refs[0](), 'exit reader survived into final audit')
                        if tamper:
                            native.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                            os.utime(native, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    with timer(context, name): yield
                collecting = driver.gc.isenabled(); driver.gc.disable()
                f.context['fit_context']['phase_seconds'].clear(); f.reads.clear()
                try:
                    with patch.object(driver, 'NEAREST', nearest), patch.object(driver, 'timed', phase), \
                            patch.object(driver, 'exit_admission_adapter', track), \
                            patch.object(driver, 'helper_guard', lambda c: f.fitter.prepare_readout(c['fit_context'])), \
                            redirect_stdout(io.StringIO()):
                        if tamper:
                            with self.assertRaises(ValueError): driver.exit_rehash(f.context)
                        else:
                            driver.exit_rehash(f.context)
                            self.assertCountEqual([r for r in f.reads if r[0] == 'origin_sha'],
                                                  [('origin_sha', p) for p in f.observed] * 3)
                finally:
                    native.write_bytes(saved)
                    if collecting: driver.gc.enable()
            self.assertLessEqual(sum(p.stat().st_size for p in f.root.rglob('*') if p.is_file()), 16 * 1024**2)
            path = Path(supplemental); saved, stamp = path.read_bytes(), path.stat()
            try:
                path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:]); os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                with self.assertRaises(ValueError):
                    f.context['nearest'].native_source_api(f.context).audit_origins(f.legacy, admission=f.original.FlatAdmission())
            finally: path.write_bytes(saved)


    def test_exit_adapter_releases_private_reader_cycles_on_rejection(self):
        with self.composition() as f:
            f.context['nearest'] = SimpleNamespace(native_source_api=self.fixtures.driver.native_source_api)
            reader = f.original.FlatAdmission()
            f.api.audit_origins(f.legacy, admission=reader)
            admitted = driver.exit_admission_adapter(f.context, f.api, reader)
            reader_ref = weakref.ref(reader)
            collecting = driver.gc.isenabled()
            driver.gc.disable()
            try:
                with self.assertRaisesRegex(ValueError, 'owned exit fitter context'):
                    admitted(object())
                del reader, admitted
                self.assertIsNone(reader_ref(), 'private exit cycles retained the boundary reader')
            finally:
                if collecting: driver.gc.enable()


    def test_post_run_promotion_reuses_only_its_fresh_origin_reader(self):
        with self.composition() as f:
            f.context['nearest'] = SimpleNamespace(native_source_api=self.fixtures.driver.native_source_api)
            f.admission.verified.update(f.observed)
            fn = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
            blocks = [n for n in fn.body if isinstance(n, ast.With) and
                      n.items[0].context_expr.args[1].value in
                      ('post_run_api_authentication', 'post_run_origin_audit', 'origin_guard_promotion')]
            self.assertEqual(len(blocks), 3)
            discard = next(n for n in fn.body if isinstance(n, ast.Delete) and
                           ast.dump(n) == ast.dump(ast.parse('del post_run_reader').body[0]))
            values = {'context': f.context, 'legacy': f.legacy, 'args': f.context['args']}
            def execute(nodes):
                exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                             str(PATH), 'exec'), vars(driver), values)
            f.reads.clear()
            with redirect_stdout(io.StringIO()): execute(blocks[:2])
            reader = values['post_run_reader']
            self.assertIs(type(reader), f.original.FlatAdmission)
            self.assertIsNot(reader, f.admission)
            self.assertEqual(reader.verified, set(f.observed))
            with redirect_stdout(io.StringIO()): execute(blocks[2:])
            self.assertCountEqual(f.reads, [('origin_sha', p) for p in f.observed])
            self.assertTrue(all(f.context['guards'][p] == h for p, h in f.observed.items()))
            reader_ref = weakref.ref(reader)
            execute([discard]); del reader
            self.assertIsNone(reader_ref())
            path = Path(next(iter(f.cpu['files'])))
            saved, stamp = path.read_bytes(), path.stat()
            try:
                path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:]); os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'unknown or changed'):
                    execute(blocks[:2])
            finally: path.write_bytes(saved)


class ContractTests(unittest.TestCase):
    def api(self, name):
        self.assertTrue(hasattr(driver, name), "missing bounded trainer API: " + name)
        return getattr(driver, name)

    def launch(self, phase="cpu", arm="control", seed=179061):
        self.api("check_launch")
        unit = {"receipt": {"path": "/unit/receipt.json", "sha256": "a" * 64},
                "log": {"path": "/unit/log", "sha256": "b" * 64}, "unit": "test-unit",
                "invocation_id": "c" * 32, "service_seconds": 1.,
                "native_peak_rss_kib": 10, "both_locks_held": True}
        value = {"schema": driver.AUTHORITY_SCHEMA, "execution_sha256": "d" * 64,
                 "phase": phase, "arm": arm, "seed": seed,
                 "nearest": copy.deepcopy(driver.NEAREST), "fitter": copy.deepcopy(driver.FITTER),
                 "accepted": copy.deepcopy(driver.ACCEPTED), "readout": copy.deepcopy(driver.READOUT),
                 "recipe": copy.deepcopy(driver.RECIPE), "resource_policy": driver.policy(phase),
                 "both_locks_held": True, "native_authority": {"path": "/native.json", "sha256": "e" * 64},
                 "selected_cpu": None if phase == "cpu" else copy.deepcopy(unit),
                 "selected_mechanics": {a: copy.deepcopy(unit) for a in driver.ARMS} if phase == "train" else None}
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256="d" * 64)
        return value, args

    def test_fixed_launch_and_seed_prerequisites(self):
        check = self.api("check_launch")
        for phase, seed in [("cpu", 179061), ("mechanics", 179061), ("train", 179069)]:
            value, args = self.launch(phase, seed=seed)
            check(value, args)
            for key, bad in [("seed", 179070), ("recipe", {}), ("nearest", {}),
                             ("readout", {}), ("both_locks_held", False)]:
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    check({**value, key: bad}, args)
            with self.assertRaises(ValueError):
                check({**value, "extra": True}, args)
        for phase, arm, seed in [("cpu", "candidate", 179061), ("cpu", "control", 179069),
                                 ("mechanics", "control", 179069)]:
            value, args = self.launch(phase, arm, seed)
            with self.assertRaises(ValueError):
                check(value, args)
        value, args = self.launch("train")
        for missing in [None, {}, {"candidate": value["selected_mechanics"]["candidate"]}]:
            with self.assertRaises((ValueError, TypeError, AttributeError)):
                check({**value, "selected_mechanics": missing}, args)

    def test_current_file_bytes_symlink_fifo_and_restored_mtime(self):
        bound = self.api("bound_file")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "member"
            path.write_bytes(b"accepted bytes")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            guards = {}
            self.assertEqual(bound(guards, path, sha), path)
            stamp = path.stat()
            path.write_bytes(b"modified bytes")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaises(ValueError):
                bound(guards, path, sha)
            link = root / "link"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                bound({}, link, hashlib.sha256(path.read_bytes()).hexdigest())
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                bound({}, fifo, sha)
            with self.assertRaises(ValueError):
                bound({}, Path("relative"), sha)

    def test_strict_json_and_exact_two_file_closure(self):
        parse = self.api("strict_json")
        closure = self.api("closure")
        for raw in ["{\"a\":1,\"a\":2}", "{\"a\":NaN}"]:
            with self.assertRaises(ValueError):
                parse(raw)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = {}
            for name in driver.FILES:
                (root / name).write_text("# tiny fixture\n")
                code[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            manifest = root / "execution.json"
            manifest.write_text(json.dumps(code))
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(closure(root, sha, driver.FILES, {}), code)
            for names in [set(), {"../outside"}, driver.FILES | {"third.py"}]:
                with self.assertRaises(ValueError):
                    closure(root, sha, names, {})
            (root / next(iter(driver.FILES))).write_text("# replaced\n")
            with self.assertRaises(ValueError):
                closure(root, sha, driver.FILES, {})

    def test_global_both_view_denominators(self):
        denominators = self.api("loss_denominators")
        self.assertEqual(denominators(25), (128, 50))
        self.assertEqual(denominators(0), (128, None))
        for invalid in [-1, 65, True, 2.5]:
            with self.assertRaises(ValueError):
                denominators(invalid)
        # Independent full-batch scalar oracle, including singleton-only micros.
        valid_groups = [16, 8, 0, 1] * 2
        rank_groups = [[.1] * n for n in valid_groups]
        batch, rank = denominators(25)
        partial = math.fsum(math.fsum(g) / rank for g in rank_groups)
        self.assertAlmostEqual(partial, math.fsum(sum(rank_groups, [])) / 50)
        self.assertAlmostEqual(math.fsum([16 / (batch * 3)] * 8), 1 / 3)

    def test_original_miner_ties_singletons_and_nonfinite(self):
        self.api("NEAREST")
        source = PATH.with_name("train_siglip2_nearest_ranking.py")
        spec = importlib.util.spec_from_file_location("compact_mining_oracle", source)
        nearest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nearest)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), driver.NEAREST["code"][source.name])
        choose = nearest.select_nearest
        self.assertEqual(choose([1., .5, .5, .5], [0, 0, 0, 1], [9, 8, 2, 4], 0), (2, 3))
        self.assertEqual(choose([.7, .7, .7], [0, 1, 2], [9, 2, 5], 0), (-1, 1))
        for scores, rows in [([1., float("nan")], [0, 1]), ([1., 2.], [1, 1])]:
            with self.assertRaises(ValueError):
                choose(scores, [0, 1], rows, 0)

    def test_complete_unit_and_resource_policy(self):
        check = self.api("check_unit")
        value, _ = self.launch("train")
        unit = value["selected_cpu"]
        check(unit)
        for key, bad in [("both_locks_held", False), ("service_seconds", float("inf")),
                         ("invocation_id", "unknown"), ("receipt", {"path": "/a", "sha256": "guess"})]:
            with self.assertRaises(ValueError):
                check({**unit, key: bad})
        self.assertEqual(driver.policy("cpu")["seconds"], 500)
        self.assertEqual(driver.policy("mechanics")["seconds"], 600)
        self.assertEqual(driver.policy("train")["seconds"], 600)
        self.assertEqual(driver.policy("train")["host_bytes"], 8 * 1024**3)
        with self.assertRaises(ValueError):
            driver.policy("quality")

    def test_updated_A_current_bytes_binding(self):
        own = self.api("own_A")
        keyword_defaults = own.__kwdefaults__
        own = FunctionType(own.__code__, {**own.__globals__, "own_residual": lambda *a, **k: None})
        own.__kwdefaults__ = keyword_defaults
        # Metadata stand-in: no Torch import. A noninitial update can still be
        # changed through a .data-like alias without a version/counter change.
        context = {"nearest": SimpleNamespace(fingerprint=lambda c, v, **kw:
                   hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest())}
        state = {"A": {"bytes": [1., 2.]}, "counter": 0}
        own(context, state, admit=True)
        state["A"]["bytes"][0] = 3.
        state["counter"] = 1
        own(context, state, advanced=True)
        own(context, state)
        state["A"]["bytes"][0] = 4.
        with self.assertRaises(ValueError):
            own(context, state)
        with self.assertRaises(ValueError):
            own(context, state, advanced=True)
        state["A"]["bytes"][0] = 3.
        own(context, state)
        state["counter"] = 2
        with self.assertRaises(ValueError):
            own(context, state)

    def test_terminal_rejects_partial_false_and_nonfinite_cpu_proof(self):
        check = self.api("check_terminal_record")
        launch, _ = self.launch()
        source, flags = {"source": "fixture"}, {"flags": "fixture"}
        bank = SmoothAPTests().bank()
        batch = list(range(12, 76))
        members = driver.ranking_membership(bank, batch)
        ident = {"method": driver.method(launch), "source": source, "arm": "control", "seed": 179061,
                 "device": "cpu", "parameter_names": ["A"], "parameter_shapes": [[128, 160]],
                 "numerical_flags": flags, "ranking_bank_sha256": bank["sha256"]}
        record = {"schema": driver.SCHEMA, "phase": "cpu", "arm": "control", "seed": 179061,
                  "launch": launch, "source": source, "identity": ident, "ranking_bank": bank, "code": {n: "a" * 64 for n in driver.FILES},
                  "authority": {"path": "/authority.json", "sha256": "b" * 64}, "authority_sha256": "b" * 64,
                  "resource_policy": driver.policy("cpu"), "optimizer_members": 1, "trainable_scalars": 20480,
                  "frozen_vision_members": 448, "quality_read": False, "total_training_core_seconds": 1.,
                  "wall_seconds": 2., "process_peak_rss_kib": 100, "numerical_flags": flags,
                  "completed_step": 0, "cuda_initialized": False, "peak_cuda_allocated_bytes": 0,
                  "invocation": {"optimize": 0, "cuda_visible_devices": ""},
                  "gradients": [{"seed": seed, "batch": batch, "membership_sha256": driver.json_sha256(members),
                                "mse": 1., "rank": .1, "active": 1, "K": 64,
                                "control_gradient_norm": 1., "ranking_gradient_norm": .2,
                                "candidate_gradient_norm": 1.1, "candidate_minus_control_gradient_norm": .2, "gradient_alignment": -.2,
                                "multi_positive_anchors": 64,
                                "nonnearest_positive_terms": 2 * sum(len(p)-1 for p in members["positive"]),
                                "nonnearest_loss": .08, "nonnearest_gradient_norm": .1, "native_mask_self_ties_singletons_exact": True,
                                "candidate_minus_control_equals_rank": True,
                                "micro16_global_reduction_exact": True} for seed in driver.SEEDS],
                  "checkpoint": {"path": "/unit/initializer.pt", "sha256": "c" * 64},
                  "bundle": {"path": "/unit/bundle/bundle.json", "sha256": "d" * 64},
                  "inference_state_sha256": "e" * 64,
                  "input_guards": {"/unit/initializer.pt": "c" * 64, "/unit/bundle/bundle.json": "d" * 64}}
        required = ("pass", "strict_reload_exact", "exit_rehash_pass", "sequential_model_ownership",
                    "forward_oracle_exact", "native_training_inference_exact", "inference_artifact_independent",
                    "bundle_original_dependencies_denied", "both_locks_held_in_parent_authority",
                    "initial_arm_parity", "cpu_serialization_exact", "bypass_version_tamper_rejected",
                    "malformed_state_rejected", "native_role_mutation_rejected", "native_loss_reduction_exact")
        record.update({k: True for k in required})
        record["gradients"] = [image_anchor_gradient_fixture(g) for g in record["gradients"]]
        record["active_objective_source"] = driver.ACTIVE_OBJECTIVE_SOURCE
        record.update(initial_C_sha256='f'*64, current_C_sha256='f'*64, mu_train_sha256='a'*64,
                      mu_train_provenance_sha256='b'*64, C_exact_zero=True, C_trainable=False,
                      residual_nonzero_witness=False, omitted_C_mutant_rejected=True, wrong_mu_mutant_rejected=True)
        ident.update({k: record[k] for k in ('initial_C_sha256','mu_train_sha256','mu_train_provenance_sha256')})
        check(record, launch, "cpu", "control", 179061)
        for key in required:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check({**record, key: False}, launch, "cpu", "control", 179061)
        for bad in [[], record["gradients"][:1]]:
            with self.assertRaises(ValueError):
                check({**record, "gradients": bad}, launch, "cpu", "control", 179061)
        for key in ("mse", "rank", "control_gradient_norm", "ranking_gradient_norm",
                    "candidate_gradient_norm", "candidate_minus_control_gradient_norm",
                    "gradient_alignment", "nonnearest_loss", "nonnearest_gradient_norm"):
            bad = copy.deepcopy(record)
            bad["gradients"][0][key] = float("inf")
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(bad, launch, "cpu", "control", 179061)

    def test_bundle_owned_files_and_forbidden_dependencies(self):
        admit = self.api("admit_bundle")
        self.api("deny_training_dependencies")
        authenticate_environment = self.api('authenticate_bundle_environment')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            code = {}
            for name in driver.FILES | driver.SERVING_FILES | {"joint_relational_compaction.py"}:
                source = PATH.parent.parent / "src/sfora" / name if name == "joint_relational_compaction.py" else PATH.with_name(name)
                shutil.copyfile(source, bundle / name)
                code[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            files = {}
            for name in ("endpoint.pt", "vision.pt", "processor.json"):
                (bundle / name).write_bytes(b"tiny admission fixture")
                files[name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            packages = {n: {"root": str(root / "installed" / n)} for n in driver.NATIVE - {"sfora"}}
            constructor = Path(packages["transformers"]["root"]) / "modeling.py"
            constructor.parent.mkdir(parents=True)
            constructor.write_text("# installed fixture")
            environment = {"packages": packages, "files": {str(constructor): hashlib.sha256(constructor.read_bytes()).hexdigest()},
                           "native_files": {}, "vision_constructor": str(constructor)}
            value = {"schema": driver.BUNDLE_SCHEMA, "code": code, "files": files,
                     "environment": environment, "vision_inventory": [], "endpoint_state_sha256": "c" * 64}
            def publish(v):
                (bundle / "bundle.json").write_text(json.dumps(v))
                return hashlib.sha256((bundle / "bundle.json").read_bytes()).hexdigest()
            sha = publish(value)
            self.assertEqual(admit(bundle, sha)[0], value)
            for bad in [{**value, "schema": "siglip2-compact-ranking-bundle-v1"},
                        {**value, "teacher": {}}, {**value, "code": {**code, "third.py": "a" * 64}},
                        {**value, "files": {"endpoint.pt": files["endpoint.pt"]}}]:
                with self.assertRaises(ValueError):
                    admit(bundle, publish(bad))
            sha = publish(value)
            (bundle / "vision.pt").unlink()
            (bundle / "vision.pt").symlink_to(constructor)
            with self.assertRaises(ValueError):
                admit(bundle, sha)
            warm = root / "warm.pt"
            warm.write_bytes(b"forbidden original state")
            # RECORD is outside every package root, but original native
            # ownership admission authenticated this exact metadata inventory.
            site = root / "installed"
            record = site / "aiohappyeyeballs-2.6.2.dist-info" / "RECORD"
            record.parent.mkdir()
            record.write_text("aiohappyeyeballs/__init__.py,,\n")
            owner_record = site / "nvidia_cudnn_cu13-9.20.0.48.dist-info" / "RECORD"
            owner_record.parent.mkdir()
            owner_record.write_text("nvidia/cudnn/lib/libcudnn.so,,\n")
            metadata = [owner_record.parent / name for name in ("METADATA", "WHEEL")]
            for path in metadata:
                path.write_text("qualified vendor metadata\n")
            # Neither a lookalike nor a training payload elsewhere in the
            # installed tree becomes runtime metadata by its path alone.
            lookalike = site / "training.dist-info" / "RECORD"
            lookalike.parent.mkdir()
            lookalike.write_text("original training input\n")
            training = site / "optimizer.pt"
            training.write_bytes(b"original optimizer")
            runtime = [record, owner_record, *metadata]
            guarded = [warm, lookalike, training, *runtime]
            guards = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded}
            proof = {"authority": {"installed_site_root": str(site)},
                     "installed_record_ownership": {"records": [str(record), str(owner_record)],
                         "owners": {str(site / "nvidia/cudnn/lib/libcudnn.so"): [str(owner_record)]}},
                     "input_guards": {str(p): {"sha256": guards[str(p)], "size_bytes": p.stat().st_size}
                                      for p in runtime}}
            proof_path = root / "native-proof.json"
            proof_path.write_text(json.dumps(proof))
            proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
            authority_path = root / "native-authority.json"
            authority_path.write_text(json.dumps({"proof": {"path": str(proof_path), "sha256": proof_sha}}))
            context = {"guards": guards,
                       "nearest": SimpleNamespace(NATIVE_PROOF_PINS={"proof": proof_sha}, NATIVE_MEMBERS={
                           'libcudnn_engines_precompiled.so.9', 'libcudnn_engines_runtime_compiled.so.9',
                           'libcudnn_graph.so.9', 'libcudnn_heuristic.so.9'}),
                       "launch": {"native_authority": {"path": str(authority_path),
                           "sha256": hashlib.sha256(authority_path.read_bytes()).hexdigest()}}}
            context['guards'].update(environment['files'])
            context['legacy'] = {'selected': {'packages': packages},
                                 'origins': {'packages': packages, 'native_files': []},
                                 'prior': {'sources': {'native_environment': {'vision_constructor': {'path': str(constructor)}}}}}
            # Authenticate only the manifest, then bind its allowances to the
            # already qualified source before installing the real deny hook.
            preflight = driver.read_json({'path': str(bundle / 'bundle.json'), 'sha256': sha}, {})
            authenticate_environment(context, preflight['environment'])
            for changed in (
                    {**environment, 'files': {**environment['files'], str(warm): guards[str(warm)]}},
                    {**environment, 'files': {**environment['files'], str(training): guards[str(training)]},
                     'native_files': {str(training): guards[str(training)]}},
                    {**environment, 'files': {str(constructor): '0' * 64}},
                    {**environment, 'vision_constructor': str(warm)}):
                with self.subTest(environment=changed), self.assertRaises(ValueError):
                    authenticate_environment(context, changed)
            # Repair the earlier symlink; both public loads must independently
            # reach their full admission while the hook is enabled. Corruption
            # aborts before the native import, so these remain stdlib witnesses.
            (bundle / 'vision.pt').unlink()
            (bundle / 'vision.pt').write_bytes(b'tiny admission fixture')
            context.update(code=code, started=driver.time.perf_counter(), phase_seconds={})
            qualify = next(n for n in ast.parse(PATH.read_text()).body
                           if isinstance(n, ast.FunctionDef) and n.name == 'qualify_bundle')
            authentication = next(n for n in ast.walk(qualify) if isinstance(n, ast.With) and
                                  n.items[0].context_expr.args[1].value == 'bundle_loader_authentication')
            for _ in range(2):
                values = {'context': context, 'directory': bundle, 'sha': sha}
                with redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(authentication)], type_ignores=[])),
                                 str(PATH), 'exec'), vars(driver), values)
                portable = values['portable']
                self.assertEqual(values['bundle'], value)
                with driver.deny_training_dependencies(context, bundle, values['bundle']['environment']):
                    with self.assertRaisesRegex(ValueError, 'fixed inference device'):
                        portable.load_inference(bundle, sha, 'invalid')
                    for member in ('endpoint.pt', 'vision.pt', 'processor.json', 'prototype_residual_readout.py'):
                        path = bundle / member
                        original = path.read_bytes()
                        try:
                            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                            with self.subTest(corrupt=member), self.assertRaisesRegex(ValueError, 'current FILE bytes'):
                                portable.load_inference(bundle, sha, 'cpu')
                        finally: path.write_bytes(original)
                    for path in (warm, lookalike, training):
                        with self.assertRaisesRegex(ValueError, 'original training dependency'):
                            path.read_bytes()
                self.assertIs(sys.modules.pop(values['portable_name']), portable)
            with driver.deny_training_dependencies(context, bundle, environment):
                self.assertEqual(record.read_text(), "aiohappyeyeballs/__init__.py,,\n")
                for path in runtime:
                    self.assertTrue(path.read_bytes())
                for path in (warm, lookalike, training, authority_path, proof_path):
                    with self.subTest(denied=path), self.assertRaises(ValueError):
                        path.read_bytes()
                self.assertEqual(constructor.read_text(), "# installed fixture")
            self.assertEqual(warm.read_bytes(), b"forbidden original state")
            for path in (record, proof_path, authority_path):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.subTest(tamper=path), self.assertRaises(ValueError):
                    with driver.deny_training_dependencies(context, bundle, environment):
                        pass
                path.write_bytes(original)
            for bad_guard in (None, "0" * 64):
                broken = {**context, "guards": dict(context["guards"])}
                if bad_guard is None:
                    broken["guards"].pop(str(record))
                else:
                    broken["guards"][str(record)] = bad_guard
                with self.subTest(guard=bad_guard), self.assertRaises(ValueError):
                    with driver.deny_training_dependencies(broken, bundle, environment):
                        pass
            broken = {**context, "nearest": SimpleNamespace(NATIVE_PROOF_PINS={"proof": "0" * 64})}
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(broken, bundle, environment):
                    pass
            # A symlink/parent alias cannot inherit an authenticated allowance.
            original = record.read_bytes()
            record.unlink()
            record.symlink_to(lookalike)
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(context, bundle, environment):
                    pass
            record.unlink()
            record.write_bytes(original)
            wrong_environment = copy.deepcopy(environment)
            wrong_environment["packages"]["torch"]["root"] = str(root / "other" / "torch")
            with self.assertRaises(ValueError):
                with driver.deny_training_dependencies(context, bundle, wrong_environment):
                    pass

    def test_completion_timing_preserves_checks(self):
        # Whole-module AST at fa2a8bb9; no general call/statement normalization.
        baseline = "d33ab2444ad0b66064c9f038524f4738544b6f03be932ff9df0bb897cbfd759f"
        reader_calls = [
            ('prepare_native', "context['nearest'].native_source_api(context).audit_origins(legacy)",
             "legacy['original'].FlatAdmission()", 1),
            ('gpu_run', "api.audit_origins(context['legacy'], require_exact=True)",
             "context['legacy']['original'].FlatAdmission()", 1),
            ('exit_rehash', "api.audit_origins(context['legacy'], require_exact=context['args'].phase != 'cpu')",
             "context['legacy']['original'].FlatAdmission()", 2),
            ('run', "api.audit_origins(legacy, require_exact=args.phase != 'cpu')",
             "legacy['original'].FlatAdmission()", 1),
        ]
        allowed_readers = {}
        for fn, text, reader, count in reader_calls:
            original = ast.parse(text, mode='eval').body
            changed = copy.deepcopy(original)
            changed.keywords.insert(0, ast.keyword(arg='admission', value=ast.parse(reader, mode='eval').body))
            allowed_readers[(fn, ast.dump(changed))] = (original, count)
        restored_readers = {}

        class RestoreAuditReaders(ast.NodeTransformer):
            function = None

            def visit_FunctionDef(self, node):
                prior, self.function = self.function, node.name
                result = self.generic_visit(node)
                self.function = prior
                return result

            def visit_Call(self, node):
                key = (self.function, ast.dump(node))
                if key in allowed_readers:
                    restored_readers[key] = restored_readers.get(key, 0) + 1
                    return copy.deepcopy(allowed_readers[key][0])
                return self.generic_visit(node)

        scheduled_inverse = completion_source_boundary(ast.parse(PATH.read_text()))
        original_calls = RestoreAuditReaders().visit(copy.deepcopy(scheduled_inverse))
        self.assertEqual(restored_readers, {key: count for key, (_, count) in allowed_readers.items()})
        phases = {
            "cpu_witnesses": {"cpu_bundle_qualification"},
            "gpu_run": {"gpu_bundle_qualification", "post_calibration_api_authentication",
                        "post_calibration_origin_audit"},
            "qualify_bundle": {"bundle_image_loading", "bundle_loader_authentication",
                               "bundle_dependency_denial", "bundle_loader", "bundle_native_forward",
                               "bundle_oracle", "bundle_release"},
            "run": {"post_run_api_authentication", "post_run_origin_audit", "origin_guard_promotion"},
            "exit_rehash": {"post_exit_api_authentication", "post_exit_origin_audit"},
        }
        splits = {"post_calibration_api_authentication": "post_calibration_origin_audit",
                  "post_run_api_authentication": "post_run_origin_audit",
                  "post_exit_api_authentication": "post_exit_origin_audit"}
        case = self
        seen, acquisitions = set(), set()

        class StripTimers(ast.NodeTransformer):
            function = None

            def visit_FunctionDef(self, node):
                prior, self.function = self.function, node.name
                result = self.generic_visit(node)
                self.function = prior
                return result

            def visit_With(self, node):
                call = node.items[0].context_expr
                name = (call.args[1].value if isinstance(call, ast.Call) and
                        isinstance(call.func, ast.Name) and call.func.id == "timed" and
                        len(call.args) == 2 and isinstance(call.args[1], ast.Constant) else None)
                if name not in set().union(*phases.values()):
                    return self.generic_visit(node)
                case.assertIn(name, phases.get(self.function, set()))
                case.assertNotIn((self.function, name), seen)
                seen.add((self.function, name))
                case.assertEqual(len(node.items), 1)
                case.assertIsNone(node.items[0].optional_vars)
                case.assertEqual(ast.dump(call.args[0]), ast.dump(ast.Name(id="context", ctx=ast.Load())))
                case.assertEqual(call.keywords, [])
                if name in splits:
                    case.assertEqual(len(node.body), 1)
                    assignment = node.body[0]
                    expected = ast.parse("api = context['nearest'].native_source_api(context)").body[0]
                    case.assertEqual(ast.dump(assignment), ast.dump(expected))
                    acquisitions.add(id(assignment))
                self.generic_visit(node)
                return node.body

        restored = StripTimers().visit(original_calls)
        self.assertEqual(seen, {(fn, name) for fn, names in phases.items() for name in names})
        reversed_splits = []

        def reverse_splits(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    reverse_splits(value)
                elif isinstance(value, list):
                    result = []
                    index = 0
                    while index < len(value):
                        child = value[index]
                        if isinstance(child, ast.AST):
                            reverse_splits(child)
                        if id(child) in acquisitions:
                            audit = value[index + 1]
                            case.assertIsInstance(audit, ast.Expr)
                            case.assertEqual(ast.dump(audit.value.func),
                                ast.dump(ast.parse("api.audit_origins").body[0].value))
                            audit.value.func.value = child.value
                            reversed_splits.append(child)
                            result.append(audit)
                            index += 2
                        else:
                            result.append(child)
                            index += 1
                    setattr(node, field, result)

        reverse_splits(restored)
        self.assertEqual(len(reversed_splits), 3)
        self.assertEqual(hashlib.sha256(ast.dump(restored).encode()).hexdigest(), baseline)
        original_exit = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "exit_rehash")
        timed_exit = next(n for n in scheduled_inverse.body
                          if isinstance(n, ast.FunctionDef) and n.name == "exit_rehash")

        def compile_exit(node):
            namespace = dict(vars(driver))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
                         str(PATH), "exec"), namespace)
            return namespace["exit_rehash"]

        with TemporaryDirectory() as directory:
            root = Path(directory)

            def fixture(name):
                path = root / name
                path.write_bytes(name.encode())
                return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

            guards = dict(fixture(name) for name in ("first-input", "second-input", "last-input"))
            supplemental = fixture("supplemental-library")
            origin = fixture("observed-origin")
            fitter = fixture("fitter-union")
            helper = fixture("helper")

            def make_closure(name, names):
                path = root / name
                path.mkdir()
                code = {}
                for member in sorted(names):
                    (path / member).write_bytes(member.encode())
                    code[member] = hashlib.sha256((path / member).read_bytes()).hexdigest()
                execution = path / "execution.json"
                execution.write_text(json.dumps(code))
                return {"root": str(path), "code": code,
                        "execution_sha256": hashlib.sha256(execution.read_bytes()).hexdigest()}

            own = make_closure("own", driver.FILES)
            nearest = make_closure("nearest", driver.NEAREST["code"])
            sentinel = ValueError("final native failure")
            real_bound = driver.bound_file

            def exercise(function, phase="mechanics", fail=None):
                trace, captures = [], io.StringIO()
                acquisitions_count = 0

                def bound(values, path, sha):
                    trace.append(("hash", str(path), sha))
                    return real_bound(values, path, sha)

                def authenticate():
                    trace.append(("authenticate",))
                    bound({}, *supplemental)

                def audit(legacy, *, admission=None, require_exact):
                    self.assertIs(legacy, context["legacy"])
                    trace.append(("audit", require_exact))
                    authenticate()
                    bound({}, *origin)
                    if acquisitions_count == 2 and fail == "audit":
                        raise sentinel

                def exit_fitter(value):
                    self.assertIs(value, context["fit_context"])
                    trace.append(("fitter_exit",))
                    authenticate()
                    bound({}, *fitter)

                def acquire(value):
                    nonlocal acquisitions_count
                    self.assertIs(value, context)
                    acquisitions_count += 1
                    trace.append(("api",))
                    authenticate()
                    if acquisitions_count == 2 and fail == "api":
                        raise sentinel
                    return SimpleNamespace(audit_origins=audit, exit_rehash=exit_fitter)

                def helper_guard(value):
                    self.assertIs(value, context)
                    trace.append(("helper",))
                    bound({}, *helper)

                bindings = {**vars(driver), "bound_file": bound}
                bindings["read_json"] = FunctionType(driver.read_json.__code__, bindings)
                real_closure = FunctionType(driver.closure.__code__, bindings)

                def closure(path, sha, names, values):
                    trace.append(("closure", str(path)))
                    return real_closure(path, sha, names, values)

                context = {"root": Path(own["root"]), "code": own["code"], "guards": dict(guards),
                           "args": SimpleNamespace(phase=phase, execution_sha256=own["execution_sha256"]),
                           "legacy": {"original": SimpleNamespace(FlatAdmission=object)},
                           "fit_context": {}, "started": driver.time.perf_counter(),
                           "phase_seconds": {}, "nearest": SimpleNamespace(native_source_api=acquire,
                               require_no_model=lambda value: trace.append(("no_model",)))}
                error = None
                with patch.dict(function.__globals__, {"bound_file": bound, "closure": closure,
                        "helper_guard": helper_guard, "NEAREST": nearest}), redirect_stdout(captures):
                    try:
                        function(context)
                    except ValueError as caught:
                        error = caught
                events = [json.loads(line) for line in captures.getvalue().splitlines()]
                return trace, error, events, context["phase_seconds"]

            reference = compile_exit(original_exit)
            timed_reference = compile_exit(timed_exit)
            for phase in ("cpu", "mechanics", "train"):
                expected, error, _, _ = exercise(reference, phase)
                self.assertIsNone(error)
                actual, error, events, seconds = exercise(timed_reference, phase)
                self.assertIsNone(error)
                self.assertEqual(actual, expected)
                self.assertEqual([row for row in actual if row[0] == "audit"],
                                 [("audit", phase != "cpu")] * 2)
                self.assertEqual([row[1] for row in actual if row[0] == "hash" and row[1] in guards], list(guards))
                self.assertEqual([row for row in actual if row[0] == "api"], [("api",)] * 2)
                self.assertEqual([row for row in actual if row[0] == "authenticate"], [("authenticate",)] * 5)
                self.assertEqual([(e["phase"], e["boundary"]) for e in events], [
                    ("source_exit_rehash", "begin"), ("source_exit_rehash", "end"),
                    ("own_exit_rehash", "begin"), ("own_exit_rehash", "end"),
                    ("post_exit_api_authentication", "begin"), ("post_exit_api_authentication", "end"),
                    ("post_exit_origin_audit", "begin"), ("post_exit_origin_audit", "end")])
                self.assertEqual(set(seconds), {"source_exit_rehash", "own_exit_rehash",
                                               "post_exit_api_authentication", "post_exit_origin_audit"})
            for failure in ("api", "audit"):
                expected, error, _, _ = exercise(reference, fail=failure)
                self.assertIs(error, sentinel)
                actual, error, _, _ = exercise(timed_reference, fail=failure)
                self.assertIs(error, sentinel)
                self.assertEqual(actual, expected)
            for path in guards:
                member = Path(path)
                saved, stamp = member.read_bytes(), member.stat()
                try:
                    member.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                    os.utime(member, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    self.assertEqual(member.stat().st_mtime_ns, stamp.st_mtime_ns)
                    for function in (reference, timed_reference):
                        with self.subTest(tamper=path):
                            self.assertIsInstance(exercise(function)[1], ValueError)
                finally:
                    member.write_bytes(saved)

            # These mutants must fail the same trace/failure contract.
            omitted = copy.deepcopy(timed_exit)
            own_timer = next(n for n in omitted.body if isinstance(n, ast.With) and
                             n.items[0].context_expr.args[1].value == "own_exit_rehash")
            own_timer.body[0].body = [ast.Pass()]
            earlier = copy.deepcopy(timed_exit)
            earlier.body[1:1] = earlier.body[-2:]
            del earlier.body[-2:]
            swallowed = copy.deepcopy(timed_exit)
            swallowed.body[-1] = ast.Try(body=[swallowed.body[-1]], handlers=[ast.ExceptHandler(
                type=ast.Name(id="ValueError", ctx=ast.Load()), body=[ast.Pass()])], orelse=[], finalbody=[])
            expected = exercise(reference)[0]
            for name, mutant in (("omit_hash", omitted), ("earlier_final_audit", earlier)):
                with self.subTest(mutant=name), self.assertRaises(AssertionError):
                    self.assertEqual(exercise(compile_exit(mutant))[0], expected)
            with self.subTest(mutant="swallowed_failure"), self.assertRaises(AssertionError):
                self.assertIs(exercise(compile_exit(swallowed), fail="audit")[1], sentinel)

    def test_no_native_import_help_and_optimized_rejection(self):
        self.api("parser")
        self.assertFalse(any(n.split(".")[0] in driver.NATIVE for n in sys.modules))
        self.assertEqual(driver.parser().parse_args(["--execution-sha256", "a" * 64,
                         "--authority", "/a", "--authority-sha256", "b" * 64,
                         "--phase", "train", "--arm", "candidate", "--seed", "179069",
                         "--output", "/out"]).seed, 179069)
        for flags in [[], ["-O"], ["-OO"]]:
            result = subprocess.run([sys.executable, *flags, str(PATH), "--help"], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0 if not flags else 1)
            if flags:
                self.assertIn(b"optimized mode", result.stderr)


class ScalarTensor:
    """Tiny stdlib broadcast fixture executing the genuine rank kernel."""
    device = 'cpu'

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if isinstance(key, list):
            return ScalarTensor([self.values[i] for i in key])
        if isinstance(key, tuple):
            return ScalarTensor([self.values] if key[0] is None else [[v] for v in self.values])
        return ScalarTensor(self.values[key])

    def apply(self, other, operation):
        other = other.values if isinstance(other, ScalarTensor) else other
        def combine(a, b):
            if isinstance(a, list) and isinstance(b, list):
                n = max(len(a), len(b))
                return [combine(a[0 if len(a) == 1 else i], b[0 if len(b) == 1 else i]) for i in range(n)]
            if isinstance(a, list): return [combine(v, b) for v in a]
            if isinstance(b, list): return [combine(a, v) for v in b]
            return operation(a, b)
        return ScalarTensor(combine(self.values, other))

    def __sub__(self, other): return self.apply(other, lambda a, b: a - b)
    def __rsub__(self, other): return self.apply(other, lambda a, b: b - a)
    def __add__(self, other): return self.apply(other, lambda a, b: a + b)
    __radd__ = __add__
    def __mul__(self, other): return self.apply(other, lambda a, b: a * b)
    def __truediv__(self, other): return self.apply(other, lambda a, b: a / b)
    def __ne__(self, other): return self.apply(other, lambda a, b: a != b)
    def sigmoid(self):
        def stable(value):
            e = math.exp(-abs(value))
            return 1 / (1 + e) if value >= 0 else e / (1 + e)
        return self.apply(0, lambda a, b: stable(a))
    def sum(self, dim):
        if dim != 1: raise ValueError('fixture only sums bank columns')
        return ScalarTensor([math.fsum(row) for row in self.values])


class SmoothAPTests(unittest.TestCase):
    def bank(self):
        target = [i for i in range(1008) for _ in range(1 if i < 12 else 6 + (i < 379))]
        self.assertEqual(len(target), 6355)
        return driver.ranking_bank(target, list(range(6355)))

    def terms(self, scores, positive, eligible):
        # The real production tensor arithmetic runs; the fixture supplies only
        # broadcasting and scalar sigmoid. Expected values below are independent.
        torch = SimpleNamespace(tensor=lambda v, **kw: ScalarTensor(v))
        with patch.dict(sys.modules, {'torch': torch}):
            return driver.smooth_ap_terms(ScalarTensor(scores), positive, eligible).values

    def test_original_image_masks_complete_positives_and_bank_digest(self):
        self.assertTrue(hasattr(driver, 'ranking_bank'), 'all-positive bank API missing')
        bank = driver.ranking_bank([0, 0, 0, 1, 2], [9, 9, 2, 4, 7])
        facts = driver.ranking_membership(bank, [0, 2, 3])
        self.assertEqual(facts['positive'], [[2], [0, 1], []])
        self.assertEqual(facts['eligible_counts'], [3, 4, 4])
        self.assertEqual(facts['valid'], 2)
        self.assertEqual(facts['eligible_sha256'][0], hashlib.sha256(b'[2,3,4]').hexdigest())
        self.assertEqual(facts['eligible_sha256'][1], hashlib.sha256(b'[0,1,3,4]').hexdigest())
        for targets, rows in [([0], [True]), ([True], [0]), ([0], [-1]), ([], []), ([0], [0, 1])]:
            with self.subTest(targets=targets, rows=rows), self.assertRaises(ValueError):
                driver.ranking_bank(targets, rows)
        for anchor in (True, -1, 5):
            with self.assertRaises(ValueError): driver.ranking_membership(bank, [anchor])
        changed = copy.deepcopy(bank)
        changed['target'][2] = 1
        with self.assertRaises(ValueError): driver.ranking_membership(changed, [0])

    def test_sigmoid_rank_algebra_self_positive_exclusion_and_ties(self):
        self.assertTrue(hasattr(driver, 'smooth_ap_terms'), 'SmoothAP kernel missing')
        # Two positives among four eligible rows: rp=1+.5, rt=1+3*.5.
        self.assertEqual(self.terms([9, 0, 0, 0, 0], [1, 2], [1, 2, 3, 4]), [.4, .4])
        # No negatives yields AP=1 even when positive scores differ dramatically.
        self.assertEqual(self.terms([99, -1, 1], [1, 2], [1, 2]), [0., 0.])
        # One positive and one tied negative: 1-1/(1+.5)=1/3.
        self.assertAlmostEqual(self.terms([99, 0, 0], [1], [1, 2])[0], 1/3)
        self.assertEqual(self.terms([99, 1, -1], [1], [1, 2]), [0.])
        self.assertEqual(self.terms([99, -1, 1], [1], [1, 2]), [.5])

    def test_nonnearest_positive_terms_have_distinct_nonzero_derivatives(self):
        self.assertTrue(hasattr(driver, 'smooth_ap_terms'), 'SmoothAP kernel missing')
        scores = [99., .02, .01, .015, -.01]
        full = self.terms(scores, [1, 2], [1, 2, 3, 4])
        self.assertGreater(full[1], 0)
        perturbed = list(scores); perturbed[2] += 1e-6
        changed = self.terms(perturbed, [1, 2], [1, 2, 3, 4])
        self.assertGreater(abs((changed[1] - full[1]) / 1e-6), 1e-3)
        # Dropping p=2 or averaging over bank rows changes the literal objective.
        self.assertNotAlmostEqual(math.fsum(full)/2, full[0], places=7)

    def step(self, bank):
        batch = list(range(12, 76))
        full = driver.ranking_membership(bank, batch)
        members = []
        for view in driver.VIEWS:
            for offset in range(0, 64, 16):
                anchors = batch[offset:offset+16]
                facts = driver.ranking_membership(bank, anchors)
                members.append({'view': view, 'batch': anchors, **facts, 'active': facts['valid']})
        return {'step': 1, 'batch': batch, 'membership': members,
                'full_valid': 64, 'full_membership_sha256': driver.json_sha256(full),
                'arm': 'candidate', 'C_before_sha256': 'c'*64, 'C_after_sha256': 'd'*64, 'C_gradient_norm': .5,
                'scale': 128, 'gradient_norm': 1., 'ranking_gradient_norm': .2,
                'A_before_sha256': 'a'*64, 'A_after_sha256': 'b'*64,
                'core_seconds': .1, 'seconds': .2, 'mse': .2, 'rank': .1, 'loss': .2 + .1,
                'preclip_norm': 1., 'active_anchors': 128}

    def test_complete_membership_rejects_nearest_only_self_and_dropped_guards(self):
        bank = self.bank()
        step = self.step(bank)
        driver.check_steps([step], 1, 1, bank)
        mutations = [lambda r: r['membership'][0]['positive'][0].pop(),
                     lambda r: r['membership'][0]['positive'][0].append(r['batch'][0]),
                     lambda r: r['membership'][0]['eligible_counts'].__setitem__(0, 6355),
                     lambda r: r['membership'][0]['eligible_sha256'].__setitem__(0, '0'*64),
                     lambda r: r['membership'][0].__setitem__('valid', True),
                     lambda r: r['membership'][0].__setitem__('batch', r['batch'][16:32]),
                     lambda r: r.__setitem__('full_membership_sha256', '0'*64),
                     lambda r: r.__setitem__('full_valid', 16),
                     lambda r: r.__setitem__('scale', 64),
                     lambda r: r.__setitem__('A_after_sha256', r['A_before_sha256']),
                     lambda r: r.__setitem__('gradient_norm', float('nan')),
                     lambda r: r.__setitem__('rank', float('inf')),
                     lambda r: r.__setitem__('active_anchors', 127)]
        for mutate in mutations:
            changed = copy.deepcopy(step); mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                driver.check_steps([changed], 1, 1, bank)
        for key in ('target', 'original_rows'):
            changed = copy.deepcopy(bank); changed[key][0] = changed[key][1]
            with self.assertRaises(ValueError): driver.check_steps([step], 1, 1, changed)
        for incomplete in ([], [step, step]):
            with self.assertRaises(ValueError): driver.check_steps(incomplete, 1, 1, bank)

    def test_old_hinge_payload_rejected_before_native_state_access(self):
        launch, _ = ContractTests().launch()
        context = {'source': {}, 'launch': launch, 'flags': {}}
        ident = {'method': driver.method(launch), 'source': {}, 'arm': 'control', 'seed': 179061,
                 'device': 'cpu', 'parameter_names': ['A'], 'parameter_shapes': [[128, 160]],
                 'numerical_flags': {}}
        saved = {k: None for k in driver.PAYLOAD_KEYS}
        saved.update(schema='siglip2-compact-ranking-v1', identity=ident, source={}, counter=0, numerical_flags={})
        with patch.dict(sys.modules, {'torch': SimpleNamespace()}), self.assertRaisesRegex(ValueError, 'payload identity'):
            driver.check_payload(context, saved, ident, 0)
        saved['schema'] = driver.SCHEMA
        stale = copy.deepcopy(ident)
        stale['method']['recipe']['ranking'] = 'candidate coefficient1; both mine; hinge sum / (2*K*.05)'
        saved['identity'] = stale
        with patch.dict(sys.modules, {'torch': SimpleNamespace()}), self.assertRaisesRegex(ValueError, 'payload identity'):
            driver.check_payload(context, saved, stale, 0)

    def test_cpu_witness_requires_positive_nonnearest_loss_and_total_difference(self):
        bank = self.bank()
        batch = list(range(12, 76))
        members = driver.ranking_membership(bank, batch)
        witness = {'seed': 179061, 'batch': batch, 'membership_sha256': driver.json_sha256(members),
                   'mse': 1., 'rank': .1, 'active': 128, 'K': 64,
                   'control_gradient_norm': 1., 'ranking_gradient_norm': .2, 'candidate_gradient_norm': 1.1,
                   'candidate_minus_control_gradient_norm': .2, 'gradient_alignment': -.2,
                   'multi_positive_anchors': 64,
                   'nonnearest_positive_terms': 2 * sum(len(p)-1 for p in members['positive']),
                   'nonnearest_loss': .08, 'nonnearest_gradient_norm': .1,
                   'native_mask_self_ties_singletons_exact': True,
                   'candidate_minus_control_equals_rank': True, 'micro16_global_reduction_exact': True}
        witness = image_anchor_gradient_fixture(witness)
        driver.check_cpu_gradient(witness, bank)
        for key, bad in (('nonnearest_loss', 0.), ('nonnearest_gradient_norm', 0.),
                         ('candidate_minus_control_gradient_norm', .2), ('candidate_gradient_norm', float('nan')),
                         ('gradient_alignment', 1.01), ('multi_positive_anchors', 0),
                         ('nonnearest_positive_terms', 0), ('native_mask_self_ties_singletons_exact', False),
                         ('micro16_global_reduction_exact', False), ('initial_losses_and_A_gradients_matched', False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.check_cpu_gradient({**witness, key: bad}, bank)
        for mutate in (lambda b: b['target'].__setitem__(12, 1008),
                       lambda b: b['original_rows'].__setitem__(0, 1),
                       lambda b: b['target'].__setitem__(0, 12)):
            bad = copy.deepcopy(bank); mutate(bad)
            rebound = driver.ranking_bank(bad['target'], bad['original_rows'])
            with self.assertRaises(ValueError): driver.check_ranking_bank(rebound)

    def test_source_inverse_retains_regression_optimizer_payload_inference_and_exit(self):
        tree = ast.parse(PATH.read_text())
        # Existing module-wide source gates remain authoritative after only the
        # exact reviewed objective/identity nodes are restored to the old AST.
        completion_source_boundary(copy.deepcopy(tree))
        fragments = {"'temperature': .01": "'temperature': .02",
                     'return 1 - rp / rt': 'return rp / rt',
                     'rtol=1e-5, atol=1e-6': 'rtol=1e-4, atol=1e-6',
                     "'native_mask_self_ties_singletons_exact': True": "'native_mask_self_ties_singletons_exact': False",
                     "ident['ranking_bank_sha256']": "ident['static_sha256']",
                     'scaler.step(optimizer)': 'scaler.update()',
                     "'ranking_bank': bank": "'ranking_bank': {}",
                     "'ranking_bank_sha256': state['ranking_bank']['sha256']": "'ranking_bank_sha256': '0'*64",
                     "row['full_membership_sha256'] == json_sha256(full)": "row['full_membership_sha256'] != json_sha256(full)"}
        source = PATH.read_text()
        for text, replacement in fragments.items():
            self.assertIn(text, source)
            mutant = ast.parse(source.replace(text, replacement, 1))
            with self.subTest(fragment=text), self.assertRaises(ValueError):
                completion_source_boundary(mutant)
        for name in ('inference_outputs', 'helper_guard', 'exit_rehash', 'prepare_native'):
            mutant = copy.deepcopy(tree)
            fn = next(n for n in mutant.body if isinstance(n, ast.FunctionDef) and n.name == name)
            fn.body = [ast.Pass()]
            with self.subTest(guard=name), self.assertRaises(ValueError):
                completion_source_boundary(mutant)

    def test_rejects_hinge_authority_and_boolean_numeric_recipe(self):
        case = ContractTests()
        launch, args = case.launch()
        for edit in ({'schema': 'siglip2-compact-ranking-launch-v1'},
                     {'recipe': {**launch['recipe'], 'ranking': 'candidate coefficient1; both mine; hinge sum / (2*K*.05)'}},
                     {'recipe': {**launch['recipe'], 'batch': True}},
                     {'recipe': {**launch['recipe'], 'temperature': True}},
                     {'recipe': {**launch['recipe'], 'clip': True}},
                     {'recipe': {**launch['recipe'], 'updates': 128.}}):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                driver.check_launch({**launch, **edit}, args)
        # Equal-in-Python bool/int replacements must also fail on values equal to 1.
        changed = copy.deepcopy(launch); changed['recipe']['adamw']['betas'][0] = True
        with self.assertRaises(ValueError): driver.check_launch(changed, args)



def image_anchor_gradient_fixture(g):
    """Prospective receipt adapter; closed target-switch/displacement fields removed."""
    g = copy.deepcopy(g)
    g.pop('candidate_minus_control_equals_rank', None)
    g['candidate_minus_control_gradient_norm'] = 0.
    g['candidate_gradient_norm'] = g['control_gradient_norm']
    g['candidate_C_gradient_norm'] = .5
    g['initial_losses_and_A_gradients_matched'] = True
    g['nonzero_C_oracle_exact'] = True
    g['omitted_C_mutant_rejected'] = True
    g['wrong_mu_mutant_rejected'] = True
    for name in ('candidate_minus_control_equals_gallery', 'regression_gradients_identical',
                 'original_active_objective_exact', 'initial_raw_unit_packed_exact',
                 'initial_gallery_scores_matched', 'tied_gradient_equals_query_plus_gallery',
                 'detached_gallery_mutant_rejected', 'frozen_bytes_exact'):
        g[name] = True
    g['regression_difference_gradient_norm'] = 0.
    g['gallery_gradient_norm'] = .3
    g['gallery_gradient_sha256'] = 'b'*64
    g['query_gradient_norm'] = .1
    g['query_gradient_sha256'] = 'a'*64
    g['arms'] = {}
    for arm in driver.ARMS:
        row = {'canonical_mse': .5, 'augmented_mse': g['mse']-.5, 'mse': g['mse'],
               'rank': g['rank'], 'loss': g['mse']+g['rank'], 'active': g['active'],
               'micro16_global_reduction_exact': True}
        for name, norm in (('canonical_regression', .5), ('augmented_regression', .5), ('regression', 1.),
                           ('ranking', g['ranking_gradient_norm']), ('total', g[arm+'_gradient_norm'])):
            row[name+'_gradient_norm'] = norm
            row[name+'_gradient_sha256'] = 'a'*64
        g['arms'][arm] = row
    role = {'query_gradient_norm': g['query_gradient_norm'], 'gallery_gradient_norm': g['gallery_gradient_norm'],
            'tied_gradient_norm': g['ranking_gradient_norm'], 'query_gradient_sha256': 'a'*64,
            'gallery_gradient_sha256': 'b'*64, 'tied_gradient_sha256': 'a'*64,
            'query_plus_gallery_exact': True, 'full64_micro16_exact': True, 'detached_gallery_mutant_rejected': True}
    g['roles'] = {k: copy.deepcopy(role) for k in ('initial:control:A','initial:candidate:A',
        'initial:candidate:C','nonzero:candidate:A','nonzero:candidate:C')}
    return g


class ImageAnchorTests(unittest.TestCase):
    @contextmanager
    def native_identity_boundary(self):
        """Run the real boundary and unchanged payload/optimizer checks without native imports."""
        class Tensor:
            requires_grad, grad_fn = False, None
            def __init__(self, value, shape=(), dtype='torch.float32'):
                self.value, self.shape, self.dtype = value, shape, dtype
                self.device, self.ndim = SimpleNamespace(type='cpu'), len(shape)
            def tolist(self): return self.value
            def __float__(self): return float(self.value)

        def typed(value):
            if isinstance(value, Tensor):
                return ('tensor', value.dtype, value.shape, typed(value.value))
            if isinstance(value, dict):
                return ('dict', sorted([(typed(k), typed(v)) for k, v in value.items()], key=repr))
            if isinstance(value, (tuple, list)):
                return (type(value).__name__, [typed(v) for v in value])
            return (type(value).__name__, value)

        def digest(value): return hashlib.sha256(repr(typed(value)).encode()).hexdigest()
        trace = []
        bank = SmoothAPTests().bank()
        launch = {k: {} for k in ('nearest', 'fitter', 'accepted', 'readout', 'recipe', 'native_authority')}
        launch['execution_sha256'] = driver.ANCHOR_ENDPOINT['source']['execution_sha256']
        historical_tree = fullfeature_source_boundary(ast.parse(PATH.read_text()))
        historical_keys = next(ast.literal_eval(n.value) for n in historical_tree.body
                               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and
                               n.targets[0].id == 'STATIC_KEYS')
        static = {k: {} for k in historical_keys}
        static.update(provenance={'encoder': {'original': True}},
                      schedules={str(seed): Tensor([0], (128, 64), 'torch.int64') for seed in driver.SEEDS},
                      classifier=Tensor([0.], (1008, 128)),
                      target=Tensor(bank['target'], (6355,), 'torch.int64'),
                      original_rows=Tensor(bank['original_rows'], (6355,), 'torch.int64'),
                      teachers={name: Tensor([1.], shape, dtype) for name, shape, dtype in
                                [('T', (6355, 128), 'torch.float32'), ('V', (6355, 128), 'torch.float32'),
                                 ('P', (1008, 128), 'torch.float32'), ('counts', (1008,), 'torch.int64'),
                                 ('e0', (), 'torch.float32')]},
                      views={view: Tensor([1.], (6355, 1152)) for view in driver.VIEWS})
        flags = {'grad_enabled': True, 'threads': 8}
        source = {'original': True}
        ident = {'method': driver.method(launch), 'source': source, 'arm': 'candidate',
                 'ranking_bank_sha256': bank['sha256'], 'seed': 179061, 'device': 'cpu',
                 'parameter_names': ['A'], 'parameter_shapes': [[128, 160]], 'numerical_flags': flags,
                 'static_sha256': digest(static), 'initial_A_sha256': digest(Tensor([0.], (128, 160))),
                 'optimizer_defaults': {**driver.ADAM, 'decoupled_weight_decay': True},
                 'optimizer_groups': [{**driver.ADAM, 'decoupled_weight_decay': True}],
                 'initial_scaler': {}, 'initial_cpu_rng_sha256': digest(Tensor([1], (1,), 'torch.uint8')),
                 'initial_cuda_rng_sha256': None}
        disk = {'schema': 'siglip2-compact-smooth-ap-v1', 'identity': ident, 'source': source, **static,
                'A': Tensor([1.], (128, 160)), 'counter': 128, 'scaler': {}, 'numerical_flags': flags,
                'cpu_rng': Tensor([1], (1,), 'torch.uint8'), 'cuda_rng': [],
                'optimizer': {'param_groups': [{**ident['optimizer_groups'][0], 'params': [0]}],
                              'state': {0: {'step': Tensor(128.), 'exp_avg': Tensor([.1], (128, 160)),
                                            'exp_avg_sq': Tensor([.01], (128, 160))}}}}
        baseline = digest(disk)
        pin = copy.deepcopy(driver.ANCHOR_ENDPOINT)
        original_code = copy.deepcopy(pin['source']['code'])
        pin['terminal_state_sha256'], pin['A_sha256'] = baseline, digest(disk['A'])
        endpoint = {'schema': 'siglip2-compact-smooth-ap-inference-v1', 'source': source,
                    **{k: static[k] for k in ('config', 'buffers', 'processor', 'head', 'means')},
                    'A': copy.deepcopy(disk['A']), 'numerical_flags': flags, 'vision_sha256': 'a'*64}
        endpoint['fixed_sha256'] = digest(endpoint)
        pin['inference_state_sha256'] = digest(endpoint)
        record = {k: copy.deepcopy(pin[k]) for k in
                  ('authority', 'checkpoint', 'bundle', 'terminal_state_sha256', 'inference_state_sha256')}
        record.update(identity=json.loads(json.dumps(ident)), initial_A_sha256=ident['initial_A_sha256'])

        def fingerprint(context, value, **kwargs):
            if value is disk:
                trace.append('complete typed state')
                kwargs['consumed'](0, 1)
            return digest(value)

        def check_tensor(value, shape, device, frozen=True, dtype='torch.float32'):
            driver.require(value.shape == shape and value.dtype == dtype and
                           value.device is device and not value.requires_grad and value.grad_fn is None,
                           'fixture tensor differs')

        native_scope = {**vars(driver), 'SCHEMA': disk['schema'], 'fingerprint': fingerprint,
                        'helper_guard': lambda context: SimpleNamespace(check_means=lambda *args: None)}
        historical_constants = {}
        for n in historical_tree.body:
            if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and
                    n.targets[0].id in ('STATIC_KEYS', 'PAYLOAD_KEYS', 'INFERENCE_KEYS')):
                exec(compile(ast.Module(body=[n], type_ignores=[]), str(PATH), 'exec'), historical_constants)
        native_scope.update(historical_constants)
        nodes = [copy.deepcopy(n) for n in historical_tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in ('check_payload', 'check_optimizer')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), native_scope)
        native_check = native_scope['check_payload']

        def check_payload(context, saved, identity, step):
            trace.append(('native check', identity is saved['identity']))
            return native_check(context, saved, identity, step)

        original = SimpleNamespace(check_payload=check_payload, INFERENCE_SCHEMA=endpoint['schema'],
                                   admit_terminal=lambda *args: record,
                                   admit_bundle=lambda *args: ({'endpoint_state_sha256': pin['inference_state_sha256']}, {}))
        finite = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True), item=lambda: True)
        torch = SimpleNamespace(float32='torch.float32', int64='torch.int64', uint8='torch.uint8',
                                isfinite=lambda value: finite,
                                equal=lambda a, b: digest(a) == digest(b))
        class Pages:
            def __init__(self, stream): pass
            def consume(self, *args): trace.append('checkpoint pages consumed')

        def clone(context, value):
            trace.append('clone admitted A')
            return copy.deepcopy(value)

        def loaded(name, path, sha, guards):
            driver.require(name == '_compact_anchor_original' and
                           path == Path(pin['source']['root']) / 'train_siglip2_compact_ranking.py' and
                           sha == original_code['train_siglip2_compact_ranking.py'],
                           'original source substitution')
            return original

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'; path.write_bytes(b'checkpoint bytes')
            pin['checkpoint'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            record['checkpoint'] = copy.deepcopy(pin['checkpoint'])
            torch.load = lambda filename, **kwargs: disk if Path(filename) == path else endpoint
            context = {'guards': {}, 'root': Path('/prospective'), 'required_guards': {},
                       'initial_A_sha256': ident['initial_A_sha256'], 'initial_static_sha256': digest(static),
                       'source': source, 'flags': flags, 'initial': static,
                       'legacy': {'original': SimpleNamespace(CheckpointPages=Pages),
                                  'quadratic': SimpleNamespace(_check_tensor=check_tensor)},
                       'old': SimpleNamespace(check_encoder=lambda value: None, finite_tree=lambda value: None)}
            with patch.dict(sys.modules, {'torch': torch}), \
                 patch.object(driver, 'ANCHOR_ENDPOINT', pin), \
                 patch.object(driver, 'INFERENCE_KEYS', historical_constants['INFERENCE_KEYS']), \
                 patch.object(driver, 'closure', return_value=original_code), \
                 patch.object(driver, 'load_authenticated', side_effect=loaded), \
                 patch.object(driver, 'read_json', return_value=launch), \
                 patch.object(driver, 'fingerprint', side_effect=fingerprint), \
                 patch.object(driver, 'clone', side_effect=clone):
                yield SimpleNamespace(context=context, disk=disk, record=record, pin=pin, original=original,
                                      native_check=native_check, baseline=baseline, digest=digest, trace=trace)

    def test_native_identity_json_binding_passes_unchanged_strict_payload(self):
        # Passing the JSON identity directly reproduces the measured tuple/list failure.
        with self.native_identity_boundary() as case:
            with self.assertRaisesRegex(ValueError, 'complete payload identity differs'):
                case.native_check(case.context, case.disk, case.record['identity'], 128)
            before = case.digest(case.disk)
            try:
                A, original = driver.authenticate_anchor_endpoint(case.context)
            except ValueError as error:
                self.fail('matching native identity/JSON receipt rejected: ' + str(error))
            self.assertIs(original, case.original)
            self.assertEqual(case.digest(A), case.pin['A_sha256'])
            self.assertEqual(case.digest(case.disk), before)
            self.assertEqual(case.trace, [('native check', True), 'complete typed state',
                                         'checkpoint pages consumed', 'clone admitted A'])

    def test_native_identity_binding_rejects_every_field_and_json_type_substitution(self):
        with self.native_identity_boundary() as case:
            pristine = copy.deepcopy(case.record['identity'])
            for key in pristine:
                case.record['identity'] = copy.deepcopy(pristine)
                case.record['identity'][key] = 'substituted'
                case.trace.clear()
                with self.subTest(field=key), self.assertRaises(ValueError):
                    driver.authenticate_anchor_endpoint(case.context)
                self.assertNotIn('clone admitted A', case.trace)
            for key in pristine:
                case.record['identity'] = copy.deepcopy(pristine)
                case.record['identity'].pop(key)
                with self.subTest(missing=key), self.assertRaises((ValueError, KeyError)):
                    driver.authenticate_anchor_endpoint(case.context)
            case.record['identity'] = {**pristine, 'extra': None}
            with self.assertRaises(ValueError): driver.authenticate_anchor_endpoint(case.context)
            mutations = [('optimizer_defaults', 'betas', [0.8, 0.999]),
                         ('optimizer_groups', 'betas', [0.9, 0.99]),
                         ('optimizer_defaults', 'amsgrad', 0),
                         ('numerical_flags', 'grad_enabled', 1),
                         ('numerical_flags', 'threads', 8.0)]
            for section, key, value in mutations:
                case.record['identity'] = copy.deepcopy(pristine)
                member = case.record['identity'][section]
                if isinstance(member, list): member = member[0]
                member[key] = value
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    driver.authenticate_anchor_endpoint(case.context)
            case.record['identity'] = copy.deepcopy(pristine)
            case.record['identity']['seed'] = 179061.0
            with self.assertRaises(ValueError): driver.authenticate_anchor_endpoint(case.context)

    def test_native_identity_projection_cannot_replace_complete_typed_checkpoint_pin(self):
        with self.native_identity_boundary() as case:
            pristine = copy.deepcopy(case.disk)
            mutations = [lambda d: d['identity']['optimizer_defaults'].__setitem__('betas', [0.9, 0.999]),
                         lambda d: d['optimizer']['state'][0]['exp_avg'].__setattr__('value', [.2]),
                         lambda d: d.__setitem__('source', {'original': 1}),
                         lambda d: d['identity']['optimizer_groups'][0].__setitem__('betas', (0.8, 0.999)),
                         lambda d: d.__setitem__('counter', True)]
            for index, mutate in enumerate(mutations):
                case.disk.clear(); case.disk.update(copy.deepcopy(pristine)); mutate(case.disk)
                case.trace.clear()
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    driver.authenticate_anchor_endpoint(case.context)
                self.assertNotIn('clone admitted A', case.trace)
                if index < 3:
                    self.assertIn('complete typed state', case.trace)
            case.disk.clear(); case.disk.update(pristine)
            terminal_sha = case.pin['terminal_state_sha256']
            case.pin['terminal_state_sha256'] = case.record['terminal_state_sha256'] = '0'*64
            case.trace.clear()
            with self.assertRaisesRegex(ValueError, 'complete typed state differs'):
                driver.authenticate_anchor_endpoint(case.context)
            self.assertIn('complete typed state', case.trace)
            self.assertNotIn('clone admitted A', case.trace)
            case.pin['terminal_state_sha256'] = case.record['terminal_state_sha256'] = terminal_sha
            checkpoint_sha = case.pin['checkpoint']['sha256']
            case.pin['checkpoint']['sha256'] = '0'*64
            case.record['checkpoint'] = copy.deepcopy(case.pin['checkpoint'])
            with self.assertRaises(ValueError): driver.authenticate_anchor_endpoint(case.context)
            case.pin['checkpoint']['sha256'] = checkpoint_sha
            case.record['checkpoint'] = copy.deepcopy(case.pin['checkpoint'])
            case.pin['source']['code']['train_siglip2_compact_ranking.py'] = '0'*64
            with self.assertRaises(ValueError): driver.authenticate_anchor_endpoint(case.context)

    def test_both_arm_receipts_zero_and_target_difference_are_authenticated(self):
        bank = SmoothAPTests().bank()
        batch = list(range(12, 76)); members = driver.ranking_membership(bank, batch)
        g = image_anchor_gradient_fixture({'seed': 179061, 'batch': batch,
            'membership_sha256': driver.json_sha256(members), 'mse': 1., 'rank': .1, 'K': 64, 'active': 128,
            'control_gradient_norm': 1., 'candidate_gradient_norm': .9, 'ranking_gradient_norm': .2,
            'candidate_minus_control_gradient_norm': .3, 'gradient_alignment': .2,
            'multi_positive_anchors': 64, 'nonnearest_positive_terms': 2*sum(len(p)-1 for p in members['positive']),
            'nonnearest_loss': .08, 'nonnearest_gradient_norm': .1, 'native_mask_self_ties_singletons_exact': True,
            'micro16_global_reduction_exact': True})
        driver.check_cpu_gradient(g, bank)
        for arm in driver.ARMS:
            for key in ('mse', 'rank', 'loss', 'canonical_mse', 'augmented_mse', 'ranking_gradient_norm',
                        'total_gradient_norm', 'canonical_regression_gradient_norm'):
                for value in (-1., float('nan'), True):
                    bad=copy.deepcopy(g); bad['arms'][arm][key]=value
                    with self.subTest(arm=arm,key=key,value=value), self.assertRaises(ValueError):
                        driver.check_cpu_gradient(bad, bank)
            for key, value in (('canonical_regression_gradient_sha256', 'd'*64),
                               ('micro16_global_reduction_exact', False), ('loss', 100.)):
                bad=copy.deepcopy(g); bad['arms'][arm][key]=value
                with self.subTest(arm=arm,key=key), self.assertRaises(ValueError):
                    driver.check_cpu_gradient(bad, bank)
        for key in ('regression_gradients_identical', 'initial_raw_unit_packed_exact',
                    'initial_gallery_scores_matched', 'initial_losses_and_A_gradients_matched',
                    'tied_gradient_equals_query_plus_gallery', 'detached_gallery_mutant_rejected',
                    'frozen_bytes_exact', 'original_active_objective_exact'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.check_cpu_gradient({**g,key:False},bank)
        for key, value in (('gallery_gradient_norm', 0.), ('query_gradient_norm', 0.),
                           ('regression_difference_gradient_norm', .3), ('gallery_gradient_sha256', 'invalid')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.check_cpu_gradient({**g,key:value},bank)
        bad=copy.deepcopy(g); bad['arms'].pop('control')
        with self.assertRaises(ValueError): driver.check_cpu_gradient(bad,bank)

    def test_historical_pin_matches_committed_training_evidence(self):
        directory = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/smooth-ap-train-candidate-179061-v1'
        if not directory.exists():
            self.skipTest('portable exact2 has no repository evidence; frozen pins remain source-bound')
        pin = driver.ANCHOR_ENDPOINT
        receipt_path = directory / 'receipt.json'
        self.assertEqual(hashlib.sha256(receipt_path.read_bytes()).hexdigest(), pin['terminal']['receipt']['sha256'])
        self.assertEqual(hashlib.sha256((directory / 'original.log').read_bytes()).hexdigest(), pin['terminal']['log']['sha256'])
        receipt = json.loads(receipt_path.read_bytes())
        self.assertEqual((receipt['schema'],receipt['phase'],receipt['arm'],receipt['seed'],receipt['completed_step']),
                         ('siglip2-compact-smooth-ap-v1','train','candidate',179061,128))
        for key in ('authority','checkpoint','bundle','terminal_state_sha256','inference_state_sha256'):
            self.assertEqual(pin[key],receipt[key])
        self.assertEqual(pin['A_sha256'],receipt['steps'][-1]['A_after_sha256'])
        self.assertEqual(pin['source']['code'],receipt['code'])
        self.assertEqual(pin['source']['execution_sha256'],receipt['execution_sha256'])

    def test_discarded_anchor_connected_identity_rejects_detached_and_wrong_target(self):
        class Scalar:
            def __init__(self, value, gradient=0.): self.value,self.gradient=value,gradient
            def __sub__(self, other): return Scalar(self.value-other.value,self.gradient-other.gradient)
            def __add__(self, other): return Scalar(self.value+other.value,self.gradient+other.gradient)
            def __mul__(self, other):
                if not isinstance(other,Scalar): other=Scalar(other)
                return Scalar(self.value*other.value,self.gradient*other.value+self.value*other.gradient)
            __rmul__=__mul__
            def __truediv__(self, other): return self*(1/other)
            def square(self): return self*self
            def clone(self): return Scalar(self.value,self.gradient)
            def detach(self): return Scalar(self.value)
            def double(self): return self
            def sum(self): return self
            def norm(self): return Scalar(abs(self.value))
            def all(self): return self
            def item(self): return self.value
            def __float__(self): return float(self.value)
            def __getitem__(self,key): return self
        fake = SimpleNamespace(nn=SimpleNamespace(Parameter=lambda x:Scalar(x.value,1.)),
            autograd=SimpleNamespace(grad=lambda loss,A:[Scalar(loss.gradient)]),
            isfinite=lambda x:Scalar(math.isfinite(x.value)),
            allclose=lambda a,b,**kw:math.isclose(a.value,b.value,rel_tol=kw['rtol'],abs_tol=kw['atol']))
        context={'anchor_A':Scalar(3.),'initial':{'A':Scalar(1.)}}
        state={'A':Scalar(1.,1.),'teachers':{'T':Scalar(3.),'e0':2.},'views':{'canonical':Scalar(0.)}}
        def readout(context,probe,features): return probe['A']*2+Scalar(1.)
        with patch.dict(sys.modules,{'torch':fake}), patch.object(driver,'raw_features',side_effect=readout), \
             patch.object(driver,'fingerprint',return_value='a'*64):
            result=driver.anchor_displacement_witness(context,state,[0])
            self.assertEqual(result['canonical_mse'],16/256)
            self.assertEqual(result['dot_gradient_displacement'],2*result['canonical_mse'])
            self.assertEqual(state['A'].value,1.)
            with patch.object(driver,'raw_features',return_value=Scalar(7.)), self.assertRaises(ValueError):
                driver.anchor_displacement_witness(context,state,[0])
            state['teachers']['T']=Scalar(4.)
            with self.assertRaises(ValueError): driver.anchor_displacement_witness(context,state,[0])

    def test_previous_stdlib_cases_preserved_by_exact_required_inverses(self):
        tree=current_gallery_test_boundary(ast.parse(Path(__file__).read_text()))
        added={'IMAGE_ANCHOR_ORIGINAL_NODES','IMAGE_ANCHOR_ORIGINAL_SHA256','IMAGE_ANCHOR_NODE_SHA256',
               'IMAGE_ANCHOR_BASE_AST_SHA256','IMAGE_ANCHOR_TEST_BASE_AST_SHA256',
               'image_anchor_source_boundary','image_anchor_gradient_fixture','ImageAnchorTests'}
        def name(n):
            return n.name if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else (
                n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None)
        self.assertEqual({name(n) for n in tree.body if name(n) in added},added)
        tree.body=[n for n in tree.body if name(n) not in added]
        changes={
            ('smooth_ap_source_boundary','tree = image_anchor_source_boundary(tree)'): None,
            ('test_terminal_rejects_partial_false_and_nonfinite_cpu_proof',
             'record["gradients"] = [image_anchor_gradient_fixture(g) for g in record["gradients"]]'): None,
            ('test_terminal_rejects_partial_false_and_nonfinite_cpu_proof',
             'record["anchor_endpoint"] = driver.ANCHOR_ENDPOINT'): None,
            ('test_cpu_witness_requires_positive_nonnearest_loss_and_total_difference',
             'witness = image_anchor_gradient_fixture(witness)'): None}
        dump=lambda n:ast.dump(n)
        keyed={(fn,dump(ast.parse(source).body[0])):replacement for (fn,source),replacement in changes.items()}
        seen={key:0 for key in keyed}
        expressions={('step',dump(ast.parse('.2 + .1',mode='eval').body)):ast.Constant(value=.3),
                     ('test_cpu_witness_requires_positive_nonnearest_loss_and_total_difference',
                      dump(ast.Constant(value='candidate_minus_control_equals_regression_difference'))):
                         ast.Constant(value='candidate_minus_control_equals_rank')}
        expr_seen={key:0 for key in expressions}
        class Restore(ast.NodeTransformer):
            function=None
            def visit_FunctionDef(self,node):
                previous=self.function;self.function=node.name
                node=self.generic_visit(node);self.function=previous
                return node
            def visit(self,node):
                key=(self.function,dump(node))
                if key in keyed:
                    seen[key]+=1
                    return None
                if key in expressions:
                    expr_seen[key]+=1
                    return copy.deepcopy(expressions[key])
                return super().visit(node)
        tree=Restore().visit(tree)
        self.assertEqual(set(seen.values()),{1})
        self.assertEqual(set(expr_seen.values()),{1})
        self.assertEqual(hashlib.sha256(ast.dump(tree).encode()).hexdigest(),IMAGE_ANCHOR_TEST_BASE_AST_SHA256)

    def test_exact_objective_boundary_rejects_witness_and_retained_node_mutations(self):
        tree=ast.parse(PATH.read_text())
        image_anchor_source_boundary(copy.deepcopy(tree))
        for name in IMAGE_ANCHOR_NODE_SHA256:
            mutant=copy.deepcopy(tree)
            node=next(n for n in mutant.body if getattr(n,'name',None)==name or
                      isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
            if isinstance(node,ast.FunctionDef): node.body=[ast.Pass()]
            else: node.value=ast.Constant(value=None)
            with self.subTest(node=name),self.assertRaises(ValueError): image_anchor_source_boundary(mutant)
        for name in ('PAYLOAD_KEYS','INFERENCE_KEYS','SERVING_FILES'):
            mutant=copy.deepcopy(tree)
            node=next(n for n in mutant.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
            node.value=ast.Constant(value=None)
            with self.subTest(retained=name),self.assertRaises(ValueError): image_anchor_source_boundary(mutant)

    def test_zero_regression_keeps_ranking_and_actual_update_predicates(self):
        case=SmoothAPTests();bank=case.bank();row=case.step(bank)
        row.update(mse=0.,loss=row['rank'])
        driver.check_steps([row],1,1,bank)
        for key,value in (('mse',-1.),('loss',0.),('gradient_norm',0.),
                          ('ranking_gradient_norm',0.),('A_after_sha256',row['A_before_sha256'])):
            with self.subTest(key=key),self.assertRaises(ValueError): driver.check_steps([{**row,key:value}],1,1,bank)
        launch,args=ContractTests().launch()
        with self.assertRaises(ValueError):
            driver.check_launch({**launch,'schema':'siglip2-compact-smooth-ap-launch-v1'},args)

    def test_historical_admission_keeps_original_role_and_rejects_pin_substitution(self):
        node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authenticate_anchor_endpoint')
        # Exercise every new pre-native admission predicate. Existing original
        # admissions have their own unchanged tests; these callbacks supply their results.
        stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Import) and n.names[0].name=='torch')
        node.body=node.body[:stop]+[ast.Return(value=ast.Name(id='record',ctx=ast.Load()))]
        scope=dict(vars(driver));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(PATH),'exec'),scope)
        pin=driver.ANCHOR_ENDPOINT
        record={k:copy.deepcopy(pin[k]) for k in ('authority','checkpoint','bundle','terminal_state_sha256','inference_state_sha256')}
        record.update(initial_A_sha256='a'*64,identity={'static_sha256':'b'*64})
        context={'guards':{},'root':Path('/prospective'),'required_guards':{'/prospective/execution.json':'c'*64,'/retained':'d'*64},
                 'initial_A_sha256':'a'*64,'initial_static_sha256':'b'*64,'legacy':object()}
        def admit(historical,unit,phase,arm,seed):
            self.assertEqual((unit,phase,arm,seed),(pin['terminal'],'train','candidate',179061))
            self.assertEqual(historical['required_guards'],{'/retained':'d'*64})
            self.assertIs(historical['legacy'],context['legacy'])
            self.assertEqual(historical['launch'],{'original':'launch'})
            return record
        original=SimpleNamespace(admit_terminal=admit,
            admit_bundle=lambda directory,sha:({'endpoint_state_sha256':pin['inference_state_sha256']},{}))
        scope.update(closure=lambda *a:pin['source']['code'],load_authenticated=lambda *a:original,
                     read_json=lambda *a:{'original':'launch'})
        self.assertIs(scope['authenticate_anchor_endpoint'](context),record)
        for key in ('authority','checkpoint','bundle','terminal_state_sha256','inference_state_sha256','initial_A_sha256','identity'):
            saved=record[key];record[key]={'static_sha256':'0'*64} if key=='identity' else 'wrong'
            try:
                with self.subTest(key=key),self.assertRaises(ValueError):scope['authenticate_anchor_endpoint'](context)
            finally:record[key]=saved
        scope['closure']=lambda *a:{}
        with self.assertRaises(ValueError):scope['authenticate_anchor_endpoint'](context)

    def test_prospective_schemas_and_both_arm_ranking(self):
        self.assertEqual(driver.SCHEMA, 'siglip2-compact-fullfeature-residual-v1')
        self.assertEqual(driver.AUTHORITY_SCHEMA, 'siglip2-compact-fullfeature-residual-launch-v1')
        self.assertEqual(driver.INFERENCE_SCHEMA, 'siglip2-compact-fullfeature-residual-inference-v1')
        self.assertEqual(driver.BUNDLE_SCHEMA, 'siglip2-compact-fullfeature-residual-bundle-v1')
        node = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'update')
        loss = next(n.value for n in ast.walk(node) if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'loss' for t in n.targets))
        for arm in driver.ARMS:
            self.assertEqual(eval(compile(ast.Expression(loss), str(PATH), 'eval'),
                                  {'mse': 2., 'rank': .3, 'state': {'arm': arm}}), 2.3)

    def test_both_original_views_regress_to_canonical_image_with_common_e0(self):
        class Vector:
            def __init__(self, values): self.values = values
            def __getitem__(self, index): return Vector([self.values[i] for i in index])
            def __iter__(self): return iter(self.values)
            def __sub__(self, other): return Vector([a-b for a,b in zip(self,other,strict=True)])
            def square(self): return Vector([v*v for v in self])
            def sum(self): return sum(self)
        node = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'loss_terms')
        # Execute the actual target/reduction statements, independently of ranking/native imports.
        statements = [n for n in ast.walk(node) if isinstance(n, ast.Assign) and
                      len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and
                      n.targets[0].id in ('target', 'mse')]
        code = compile(ast.Module(body=statements, type_ignores=[]), str(PATH), 'exec')
        teachers = {'T': Vector([2., 5.]), 'P': Vector([3.5]), 'e0': 2.}
        for arm in driver.ARMS:
            for values in ([2., 5.], [3., 4.]):
                state = {'arm': arm, 'teachers': teachers, 'target': Vector([0, 0])}
                scope = {'state': state, 'raw': Vector(values), 'index': [0, 1], 'rows': 128}
                exec(code, scope)
                target = [3.5, 3.5]
                self.assertEqual(scope['mse'], sum((a-b)**2 for a,b in zip(values,target))/(128*2.))


class GalleryDual:
    """Four independent scalar derivatives: two query, two gallery coefficients."""
    def __init__(self, value, derivative=(0., 0., 0., 0.)):
        self.value, self.derivative = value, derivative
    def __float__(self): return float(self.value)
    def __add__(self, other):
        other = other if isinstance(other, GalleryDual) else GalleryDual(other)
        return GalleryDual(self.value + other.value, tuple(a+b for a,b in zip(self.derivative,other.derivative)))
    __radd__ = __add__
    def __neg__(self): return GalleryDual(-self.value, tuple(-v for v in self.derivative))
    def __sub__(self, other): return self + -other if isinstance(other,GalleryDual) else self + (-other)
    def __rsub__(self, other): return -self + other
    def __mul__(self, other):
        other = other if isinstance(other,GalleryDual) else GalleryDual(other)
        return GalleryDual(self.value*other.value,tuple(a*other.value+b*self.value for a,b in zip(self.derivative,other.derivative)))
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = other if isinstance(other,GalleryDual) else GalleryDual(other)
        return self * GalleryDual(1/other.value,tuple(-v/other.value**2 for v in other.derivative))
    def __rtruediv__(self,other): return GalleryDual(other)/self
    def __gt__(self,other): return self.value > float(other)
    def __ne__(self,other): return self.value != float(other)
    def sqrt(self):
        root=math.sqrt(self.value)
        return GalleryDual(root,tuple(v/(2*root) for v in self.derivative))
    def sigmoid(self):
        e=math.exp(-abs(self.value));value=1/(1+e) if self.value>=0 else e/(1+e)
        return GalleryDual(value,tuple(v*value*(1-value) for v in self.derivative))


class GalleryTensor(ScalarTensor):
    dtype = 'float32'
    device = SimpleNamespace(type='cpu')
    @property
    def shape(self):
        return (len(self.values),len(self.values[0])) if isinstance(self.values,list) and self.values and isinstance(self.values[0],list) else (len(self.values),) if isinstance(self.values,list) else ()
    @property
    def requires_grad(self): return any(isinstance(v,GalleryDual) for v in self.flat())
    @property
    def grad_fn(self): return object() if self.requires_grad else None
    def flat(self):
        def walk(v):
            if isinstance(v,list):
                for x in v: yield from walk(x)
            else: yield v
        return list(walk(self.values))
    def __getitem__(self,key):
        if isinstance(key,GalleryTensor): key=key.values
        if isinstance(key,list): return GalleryTensor([self.values[i] for i in key])
        if isinstance(key,tuple): return GalleryTensor([self.values] if key[0] is None else [[v] for v in self.values])
        return GalleryTensor(self.values[key])
    def apply(self,other,operation):
        return GalleryTensor(super().apply(other,operation).values)
    def sum(self,dim=None):
        return GalleryTensor([sum(row) for row in self.values] if dim==1 else sum(self.flat()))
    def mean(self): return self.sum()/len(self.flat())
    def square(self): return self.apply(self,lambda a,b:a*b)
    def norm(self,dim):
        return GalleryTensor([(sum(v*v for v in row).sqrt() if any(isinstance(v,GalleryDual) for v in row) else math.sqrt(sum(v*v for v in row))) for row in self.values])
    def __gt__(self,other): return self.apply(other,lambda a,b:float(a)>b)
    def all(self): return GalleryTensor(all(self.flat()))
    def item(self): return float(self.values) if isinstance(self.values,GalleryDual) else self.values
    def detach(self): return self.apply(0,lambda a,b:float(a))
    @property
    def T(self): return GalleryTensor([list(row) for row in zip(*self.values)])
    def __matmul__(self,other):
        return GalleryTensor([[sum(a*b for a,b in zip(row,column,strict=True)) for column in zip(*other.values)] for row in self.values])
    def sigmoid(self):
        return self.apply(0,lambda a,b:a.sigmoid() if isinstance(a,GalleryDual) else GalleryDual(a).sigmoid().value)
    def to(self,device): return self


class CurrentGalleryTests(unittest.TestCase):
    @contextmanager
    def numerical_roles(self, tied=False, detached=False):
        from contextlib import nullcontext
        bank=driver.ranking_bank([0,0,0,1,2],[0,1,2,3,4])
        features=GalleryTensor([[1.,.2],[.9,.3],[.8,.4],[.7,.5],[.3,1.]])
        A=[GalleryDual(.05,(1.,0.,0.,0.)),GalleryDual(-.04,(0.,1.,0.,0.))]
        G=A if tied else [GalleryDual(.05,(0.,0.,1.,0.)),GalleryDual(-.04,(0.,0.,0.,1.))]
        def raw(context,state,features):
            a,b=state['A']
            return GalleryTensor([[x+a*y,y+b*x] for x,y in features.values])
        def normalize(tensor,dim):
            return tensor / tensor.norm(dim=dim)[:,None]
        teachers_raw=raw({}, {'A':[.05,-.04]},features)
        teachers={'P':GalleryTensor([[.85,.32],[.75,.472],[.35,.988]]),'T':teachers_raw,
                  'V':normalize(teachers_raw,1),'e0':2.}
        query={'A':A,'arm':'candidate','views':{'canonical':features},'device':'cpu',
               'ranking_bank':bank,'teachers':teachers,'target':GalleryTensor(bank['target'])}
        gallery={**query,'A':G}
        torch=SimpleNamespace(float32='float32',tensor=lambda v,**kw:GalleryTensor(v),
            isfinite=lambda t:t.apply(0,lambda a,b:math.isfinite(float(a))),autocast=lambda *a,**kw:nullcontext())
        functional=SimpleNamespace(normalize=normalize)
        with patch.dict(sys.modules,{'torch':torch,'torch.nn':SimpleNamespace(functional=functional),
                                    'torch.nn.functional':functional}),patch.object(driver,'raw_features',side_effect=raw):
            if detached:
                genuine=driver.ranking_gallery
                with patch.object(driver,'ranking_gallery',side_effect=lambda c,s:genuine(c,s).detach()):
                    yield query,gallery,raw
            else: yield query,gallery,raw

    def test_candidate_rebuilds_gallery_from_current_same_A(self):
        self.assertTrue(hasattr(driver,'ranking_gallery'),'connected current gallery API missing')
        with self.numerical_roles() as (query,gallery,raw):
            first=driver.ranking_gallery({},query)
            query['A'][0]=GalleryDual(.15,(1.,0.,0.,0.))
            second=driver.ranking_gallery({},query)
            self.assertNotEqual(float(first.values[0][0]),float(second.values[0][0]))
            self.assertTrue(second.requires_grad)
            query['arm']='control'
            self.assertAlmostEqual(float(driver.ranking_gallery({},query).values[0][0]), float(second.values[0][0]))
            self.assertTrue(driver.ranking_gallery({},query).requires_grad)

    def test_real_loss_tied_gradient_and_detached_mutant(self):
        def run(tied=False,detached=False,micro=4):
            with self.numerical_roles(tied,detached) as (query,gallery,raw):
                total=rank=GalleryDual(0.)
                for view in range(2):
                    for i in range(0,4,micro):
                        anchors=list(range(i,min(i+micro,4)))
                        descriptor=raw({},query,query['views']['canonical'][anchors])
                        mse,ranking,_=driver.loss_terms({},gallery,descriptor,anchors,3)
                        total+=mse.values+ranking.values;rank+=ranking.values
                return total,rank
        split,split_rank=run();tied,tied_rank=run(tied=True);micro,_=run(tied=True,micro=1)
        self.assertAlmostEqual(split.value,tied.value,places=12)
        for j in range(2):
            self.assertAlmostEqual(tied.derivative[j],split.derivative[j]+split.derivative[j+2],places=10)
            self.assertAlmostEqual(tied.derivative[j],micro.derivative[j],places=10)
        self.assertGreater(math.hypot(*split_rank.derivative[2:]),1e-6)
        detached,_=run(tied=True,detached=True)
        self.assertGreater(math.hypot(*(a-b for a,b in zip(tied.derivative,detached.derivative))),1e-6)
        # Both arms regress to P[label]; changing candidate to T must break this oracle.
        with self.numerical_roles() as (query,gallery,raw):
            anchors=[0,1,2,3];descriptor=raw({},query,query['views']['canonical'][anchors])
            candidate=driver.loss_terms({},gallery,descriptor,anchors,3)
            control=driver.loss_terms({}, {**gallery,'arm':'control'},descriptor,anchors,3)
            self.assertEqual(float(candidate[0].values),float(control[0].values))
            for j in range(2):
                self.assertAlmostEqual(candidate[1].values.derivative[j],control[1].values.derivative[j],places=10)
            self.assertAlmostEqual(float(candidate[1].values),float(control[1].values),places=12)
            singleton=driver.loss_terms({},gallery,raw({},query,query['views']['canonical'][[4]]),[4],0)
            self.assertEqual(singleton[1].item(),0.)
            self.assertEqual(singleton[2]['active'],0)

    def test_active_objective_authentication_reads_source_only_and_rejects_substitution(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);code={}
            for name in driver.FILES:
                body="def loss_terms(*args): return 'original active'\n" if name.startswith('train') else '# exact historical test\n'
                (root/name).write_text(body);code[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            (root/'execution.json').write_text(json.dumps(code))
            source={'root':str(root),'execution_sha256':hashlib.sha256((root/'execution.json').read_bytes()).hexdigest(),'code':code}
            pin={'source':source,'checkpoint':{'path':'/nonexistent'},'terminal':None}
            context={'guards':{}}
            with patch.object(driver,'ACTIVE_OBJECTIVE_SOURCE',source):
                original=driver.authenticate_active_objective(context)
                self.assertEqual(original.loss_terms(),'original active')
                self.assertEqual(set(context['guards']),{str(root/n) for n in driver.FILES|{'execution.json'}})
                stamp=(root/'train_siglip2_compact_ranking.py').stat()
                (root/'train_siglip2_compact_ranking.py').write_text('def loss_terms(*args): return "substituted"\n')
                os.utime(root/'train_siglip2_compact_ranking.py',ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                with self.assertRaises(ValueError):driver.authenticate_active_objective(context)

    def test_precise_prospective_inverse_preserves_historical_source_and_test_guards(self):
        source=PATH.read_text();tree=ast.parse(source)
        restored=current_gallery_source_boundary(copy.deepcopy(tree))
        self.assertEqual(hashlib.sha256(ast.dump(restored).encode()).hexdigest(),CURRENT_GALLERY_BASE_AST_SHA256)
        completion_source_boundary(copy.deepcopy(tree))
        current_gallery_test_boundary(ast.parse(Path(__file__).read_text()))
        for before,after in (("target = state['teachers']['P'][state['target'][index]]",
                              "target = state['teachers']['T'][index]"),
                             ('return F.normalize(raw, dim=1)','return F.normalize(raw, dim=1).detach()'),
                             ('@ ranking_gallery(context, state).T',"@ state['teachers']['V'].T"),
                             ("'gallery_gradient_norm': roles_record['initial:candidate:A']['gallery_gradient_norm']",
                              "'gallery_gradient_norm': 0."),
                             ('scaler.step(optimizer)','scaler.update()'),
                             ("'temperature': .01","'temperature': .02")):
            self.assertIn(before,source)
            with self.subTest(mutation=before),self.assertRaises(ValueError):
                completion_source_boundary(ast.parse(source.replace(before,after,1)))
        baseline=restored
        smooth=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='smooth_ap_terms')
        self.assertEqual(ast.dump(smooth(tree)),ast.dump(smooth(baseline)))
        update=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='update')
        self.assertEqual(ast.dump(update(fullfeature_source_boundary(copy.deepcopy(tree)))),ast.dump(update(baseline)))




















FULLFEATURE_ORIGINAL_NODES = (
    'c-qZ<i*g%BlHjlOIN}1`5(tvAtdZft#9?I0r?W?9NVGRLu!xQ(&?I5vHE4hoN5_A^d{tFeb#()j?Y*1Zvxu&DRaREM^Yd53Hy7`IJ<EUo$NxU#'
    'zdru_%ZE2-U-g%}`Mmqp@Ry7K_49{!7yp@meDlA~{yF@rU;C5{l1??dp3l}VjwY+cx|nQ_CcDk1TyBr9i}}3VJRDVv)oOclRIHEY#cnydIl4RT'
    'eAWHxAHTf+{--n7wfFmJ{a)^t(|IZVzI*@n?8DjnH)r{eXaD&TI`%(6&!0Qh<fdE{oh}Tr+HEE!{xw-GuV&Zy*X8c&s@zog=X$f6lvTCb;E$WK'
    'nBu=f{Li8+mQ-cATa=sGq?qSd^WvI5-OZ}mYMECz#fxuV;fJf)r*dk3KYEk2>kmIYZ3|{gm~y$Cm|Zv@{^w6W55G5ipnnECFq=Y;v+aYw8@R@2'
    'A1{U%@7_=&__SJYXN%dtOWKPH7R-N5R=ed^{#@_!&64(TvYQs>?{J5313sL+d3O$wCV%jp%5qwD`pJA&ZPSluXWxI!p!H^Tj~`!s{mnPn@4Tq0'
    '5<i@toV><~%$C>la=Tj6w-@kbx1JVTuJQtFTo&8O4fgO7k8L*Dtjw3wS9DnAy&d(}cW3`Wqi#3FY*}2+%Y0cZsIw95$=EA_^IUV$h(<bnbutcW'
    'z@m!{R(Wyq^7RRn71LsI55GS@dp7)j_)nZ(xh?GY_-!Pg#@Qn+c(%r=_OSJAIor-)A7%Gxg;cO9uQvdv0Cn)Oa|u9^6q`kroR8+kWjP;zm(aJ>'
    'W_Arzn<upBWU^Xqrto`PCe?0{949IC_pCh0@YFWN@)q{i@9(mh+};<PDKsswu4a=Nz|QG+34mj=UR5x;yR!N&`AA?nJU6|*c=jW8xh)s#a#L(~'
    'n-aP`>7AU?#{ysx7t<*wlXAT+q4#3BS|Y?G7vCj|a&cL10KzBpT?GRq=iepE)n-x50T|PZjMsLAOG?TUx|^_1Jad>B0*>ks!Rb4g{&Krn&4YfR'
    ')^vuzlgLF&DmeV3Ay!|O6MtqKzy_<`_PeC~1QVN<RM+kBY*Ad7yff5^xn_EwTxWa^^8`TW4p2`q=R0;&PIq(MtB&pWYPPAia7@WRUcF3UHxs-y'
    '-z9J;i=$$<y}?l?umt$FD5~2k*)4ZfNq9?U>~;6%{FftGso92Ua$PQWvt^lFm2^%p%eUuWzrc0CVs_irq^P#ZbXFmJQ?FN>)xS#uJH%3O70U$h'
    '&}v$4UcP*N%vkkUQ0lS8r^kp)kLfngu4d)t*y2;cZPcMXk4*_3Rol&OvaQr<NZq8o!~fFVK7ExGd~V5lQywugNQMF=aOR~0V%#n0dR`V4Od7y2'
    'x!x4(o9~hpY=A{~NwqE4NpS^8E-7c%H(MU4LMVjY-^|R;$;1gK5`@D7;0TYm#pb%);&j#pKtEj4<T+0V4h?tq$iPHWK}!H#Q%*2_wc6lpkL|Rd'
    '%X;1Rn0<&;byyqGu6=F3<c;B%MLi1GkNX9krTAdpmCIuJ)Hl^Zg7>(>hQXylwAOV%o70nksDn0EUln<NODSj6Qgt$zx4s-XEEKjoUtRuI(qFKW'
    'X?c~j)=3d*l%KX)|G$<A{O`ZM!v8l^O%R|Q27q;LSKEhmd9((^bwo6P+114Mn(&(oAUA*L{nwJazJPuv^A+4%#v@0tp*Jf4zvXnjnk~1-Z~@oz'
    'Vp1*;N!`!3%MxEjfaL`|J;@C`B`bPyzvKR3UJro4OMqB{!%g|G9ndK7N_G=@QxHTdn7rz3bp;J)6(Hd4&1%}ytLDFX>fPje2e4M*`48W}0s1gM'
    'd;k6U&+pz}j0{Bq&!X(!sKd44dE-tt0AA4ayx88PTzk~PZ?*#B5~zUt0SGC@vTkB(mY>SWZmXy*80qc1pUys-S(#CA#Tz_B>}+7$<8OM_?XmH('
    's{<-O!KW91^B96{hkzcJ$$V8za}V#Pom{Sxg&^q~K#~@FEF?lMSI^d$C%3moB!V*=?XNht3l$iL(Y6l?p6PcAH>EOwYxV+xp;$ho>9PmVG~1?~'
    'UMCx!i~%H)_d{d|5TPVX{9Qd%y~T>IScdcw5DmCDv&DL~A&{DsQ-IcDx|lH;qE1%>2pw~tU(J9z%ClZ^S*_-~ZJB00xazQn-Q@FQ+qvU9df}nR'
    'kDR#;=mFoo{I_bgOg{rc*&^i!PYG>F&+ZT|N<gTe_(d|xW6(dKmdS0-qon)~&rdC#*32`fmA^k5emMJhVIf>TzkPRcarVJ}4PcgL0}uClx9!cU'
    '-0c~h-ZasryoWt_wpa5ztN@T+FQ;aF`l_t_1X!a=vn*Z=16L1=`D}S>ssJWSW7Xh6CE&=nt8}e;TZe_J9dwDHwnk8+eGA9Hk5B3x8`OFoo%Y8B'
    'z7;>kB@=H%Rq{m4rcXN|;t((lPOzKU4>43^3k{wT2*QJ@l?f>Ya{^KWQPHx5^QyLAhqy<+18lXTAF0-9fW0~%k_J8=3Z^;Q-)7u^d!=fRp&hqC'
    '{=!o<g~K;*&dx8+zQ<zoH*AgF6U!sm>siOl{&42x{7cqN%<uRU!ex<T3yW%}b~W3=U$sNy&v0gDUae=xwL5YAfF}nP>={@}18zVyo9>D^ADutd'
    '(E)A3ql2|A60B7jRR3{?ME-kZieaLq5CTAnZFg(6IQf#>2Lb*n;M_JTuPhkHzP<y~mqWj4Fk~7Y*r4qsZ8rnjNdr=&4r6TG7J3s97l#;s5UcIF'
    'Uf%%UFu+~yAmU#PkluxZgkOwI1|Ejvm0I{!D}_DU=CCOX+=A5XMK^K7n?eRS8qf?DJ{HnHoI1WwL@sv5`Zfh6gl26cGf)nb+m?18{q_x&x|)Nh'
    '$ytK6b{P;Ycas~$%x1lgGVB3mx`3xh=oP&@uF|W@Ezmc`ZoW;i8@iY`R)EE@g>$%?&n9+@gaRd56P}Db^dw_iAzRE3wV6g=K$Nz6iu@e@1HS|o'
    '`iXh;I|kGC4gB*dZAq+Q0G_U&%mF^TWA4n$>tgbtzKzT$qD<#guLysSov^MCdpp^tuH2B^nL3Xg8R!l6W4<7Bh47nK$Xe5KKR*)B#AjRkkWa2R'
    'tKHguC+ID)%U|BVyU2kN`~bl5=)F)qCKGd>QeLcQmWlF?1TdRCySae@Jf(FXAVh<ToW6iZGP{De{V#(UH7E0c8dN|_#i}Ty>?pVj?w-u2D@O~P'
    'S<Xq+kYl?s%Q)rop#TZ~Bta!m%~@<j084%^^_aLK<osbl!w>)bs7jtRqeVHnDVDPdY}N?68x!AbJ`qByO^*8SCYMN++O2mdJ|11UF8ZHM<H!DI'
    '=<E^BB$#<tzn{g(3Mq#k_PC4|5b|*)o8ms-!QVOH`CBJ34?AeAM&uy&XJp!luK`ERCbh1bYh=-G%-AC1ZtJh9>Yk28Vt!E1VO_lI3zTXD>a{1>'
    'z_QB01G~r^3{LZUQ%q5oD|}<UAiV~#0TR#PWERRIdn1HjE?;hD>l7=(13wbc`#5-_=H)U~g}BaFFKKV!tLw(w2xQ;nLn>8dJCQoU?obYjz0H7B'
    'S#G7!en*gHm4e^u$6Aw<<ZlTV;J7l@sB>4$XV}l*aI!%s%;z`tWJ<B=i{x2iMz#EnbaRXCmq|K9F8B(lL+J<3^#-_`D?EHt4=>^^)$S~+(unI7'
    'VzLL9qik2n^XmmxjM!EL7-CoDjh)*MuuIoVluD7}2L-jv9smtq1)uy*vdkSBE@_<neUO}}UBZPYe@{;I+W+4!`b(EbvnFsjm5)cK_zQ|6`{-3P'
    '5VruzXO<hWan9nwDhB>dHE}BssD+}k-6Ews3;g!qlT*i1+q+@fq39x*sM-+QJ3ctkxwcfsiQwg-$t;RbW<p2FX%`oA(4=q(CP+{<0+i4ED&Nk^'
    'DtAi6oTPn-FRJ8XvnyT8Ca?gc25)C>r2mnYPB~|RqRJVqPmrXqx3H4#PzD{Jp1U9NS>KVSN0WO}N<~>gjyed~6M0TZjy^-e%z&mW_+LA$jH2ya'
    'X+Eaq7G>^J+uLHdEuhh+{Ees<r!mSeAGW30Z}0ZA8YjP=uP$-k<#ab8rY>B&QLM904E152F6GVnFNvA7k!m-|3T6NSKn5&<yS(mQm)o?1>dp=f'
    'fy8xY9w+)Z?%kFT6${sZ4@rp<>av!UG&+b3j_t@yc~>>Tk_83Qn^;v~u2i{5!0)aR3lIgbzX!iWSBq!0+5$`|HjC)M&FY@$XS;+thb3D2U^eo)'
    'xM|3T;MEcQ_bf$idGb8z1n7?i$ykQ7Fe}d@1SHH<?eFJjGOc*uLLoHtPW%s{0HgSChN{QlIfiK*_#m5_1it)3o-TU{Wer-UsQ?b!G%gLOdf;xj'
    ';il|e?dEg3m}zHpa`bI+bT$6`>g6M87TS8rnhroOh%`<#9eJXNhBJR|%6YzB%?GFD(aAT!UR+l8=PN_m0`L=$CCm}3lWceOXPA)sbz`a8Qikey'
    'a~!k|aAW-rQ5OwFnpY+ha#iEIL-Yx6d2m(6Fct36yD$fBUws_9Rc0Ws`zIU<H+}+6^+b|RPi^0z5VbYoREumz79hrE8NAQ@w|c=M4Vq!v9F}Qf'
    '-skp4`X2l)w43Sc?44b79k+a&Vza&7dywim|1#8EqstUc!pDcUWA!@}puZ?h)7&0U=;Ze?Ix^m^(5q@A41rN2%b75Vliwlky{64S48ztqui6Zp'
    '!Cx>3Dac~@7B(|BV~2brsPW0@1X+vde(^Aeu{`^RW&Y>i<z{6JQjXB7L8;O$=5R0*QMi$m0<n=5Q7m>jYn_~Vi7`^;3f9WS6JXe{X;s{z$dicf'
    'E*bO~(nr)9+tp@rBf5w+iioMjNEtnjee4DWnoc1hQYj`Y#UdA3DU%h*t3dscU&fM;QwH%!@#thMpM?KGIUd*keA`d%7!<qk2MW=n?X{%zmY!zm'
    '?-Bc>-B3%^Ei*MOUYaI*)6XZ5*e6*=mGi13LO5g9>S~TAbBJT7%7|)pzhtpa_*_UgiV|fO4$eT24s)s#i(E4}hjQH^h2c6c?yl9JdG#+j4{kQ%'
    'y5k<%aniD6)j|0b@fMa7kn{*+ef^@LZh1GGls(ge#KlsR+6DEQi9m2fo7-uz_zyNWA?Ag4H`pyh)x(k*3B}i$1w!S!V!lIT3K<<*kNpZ}N2#LP'
    '5o6fOYv9uOemGq`*ODG9LL#yST!J6+t0mGL?^jFVVYx4h{i$ZwLwubP6sOGYVIf;24wJ=dQPx#UW+ifg#Z;CqP)eE-0sFv~7LXgl|NZs`Q4Mg&'
    'n*p5*!l$(0v*oVHz`4V)-_T)Z&7@%ykHt*d&DJ8fb@70zU=MT2Hz)mn3Ig+XKwS>c2psp^ZV}mn)4IeikWNdAodDV*jm)cH6i!%dwF>&9Y>-t<'
    'aclX?)@4T8ihn!RgweWYeiK4sCG4gcXESz0`|PHAVyev*m9@5CzaXN;C^x*KxD+`A45rXTLxce0kzKM~#$Z0=M8aZ=8mu&#f0I1$#cRJ+Xk)MU'
    'Jz8e>SiWUlWxy-omsq3ahC2rL@w(isH?Rl?s`xnZtZv7STvCRjkC_IFx_ZV=&E6=#j<dN4Esoi$*qD7Dv-z#H4YgL?BzclGg-bFN7wRy~=8B!^'
    'xCgWjTIpw9OCES8sjfoi>;%+Wb>Dh5wSAY0q;U!N-tG?RfqRTEWc_Bs$_f`~2q~=TVIq}d+fODEs*X06bl<~i0#Gbi4ySrFT5TkGiz(VdPYhjU'
    '>-TbA(zaYyE4w1{y>PFQJ%FKobGr!x#en!&f2G0W;@bKzjJ(q^pjJ)Pfkrf%hAKkVho6VN{r#n-zsw0}kJ*-e3NK%^yR6_|lC4Y3v;wh}#Z?h2'
    '3WfODYI;0E9XdA4#^LL4)Lre30-^vGOa3;<P9;i8i=GRnhuF*!p~LRr47v&T3F{c8zJYkk&Oe9+N#v=>5R}=b&N4hDzadwP=EZeFbMl6&W3_*V'
    'jN+}pxhgWM34*DSU6YVdb&!?i4(Pw}n`-UZ#~#UhfzQpa_@8s@kVV?4@}%>q(Tizc<GF&Jmo2}rn?gHHoAztu+qr)t^vzC!3-}{>=KykS-cq&1'
    'KWTft!^PW5!XTpPc*=ZfL!jb$+?4Y*@=0ihVl7z&1(VN$IVpBv7)Q~RT}b?p*?S>WGO%NslVRMcTPnwK`i<};cdZoqT^o-=<a?ZSuaoTvHN<=d'
    'Gp94CgdH^gM99#v*IE4CHPnx|5Y`Y1bh3S|@sO*zoX!@Cm2Shgwowd=H`r?+_imaSuSRa4o*uv*oMhgyH7@H;ts~r7HEsjt`4>AVG;Q4he+l6h'
    '#0*QQrK7drB#+<x^8GOR@cxGyak5|T5~X7JVFu2t6i9IF5jLtk=7_$aT)XPB4JA5D>~H{Wp~7`Na*m1{6@T0z(r=lS+Tgd0iK3k`oF7^4oz7{)'
    '*5&sVojj3$Fz7X`gQrISB+t1Az7uvUFX0=yP0)&>SXmTz*uS<Q!&kM%^oC^qzVS_Hfa6tBCOGsR`_GNV3zXq+X95!_R&?c<a)rtdFzdm3nQlnT'
    'GuLzc4wZy*W@pPl#l!rvgw0dtt8hJ}g#7rzD~^^rtj(cTVi8G}7d#X5MWF=NtJr19E}ChM7K{%CNEUL4ehcAStw<j#KTBQ$tSJ2ve+#%Xgc5cR'
    '6c7;Fut6hZfN9n>vNf1ZNo--UEh8%lO%mB0BpG;sHJT;=17`aS2=ac)e?Lb<75orLm&#9TcGzrzSs5gY%cUZcKH6UE->sFs&&+tfxKHLyPBF3~'
    'Ix7u?$h;aMnc`k;hgk@<ro`j)&4|s0VOA99nTMIq>|OKD-Lq~m)awyB2qtbjV~20~pg<d1Hym;}Agp0cffj1c&9I$izgwf;Y~*Ilt!la-bVQp2'
    'j<L2}`aYB25C55Yh$gf*;{#HPR&9c%G_$xRO}bgphKR({yPlrp8y^QmaQ&z$L`6z~+EJhx90^TjW4T(R9{r##-_wU#duug9#3ZK$R_}=TV64eL'
    'l_*CdXAmGh-1BbD%Xx3ShdwAv`Jb+}yC)lKv2-RjwErZp6}!*WZKE926w_#2%EsNiB~n@#g(v)x`9&f-Buc)VTD<{@I!zrb=~QA+{+0#3p>rQu'
    ')2~<a*+itW{C7(83L7~6j@(aqQiE#ss1<B<hk+}Cs$#$;VNSeWU%q@DR7Pttbg4lvOg1!lTL4|G(1X;Bo3E`4zUtL>GeghUlH8`O@ml$5h96<w'
    'H@U3}=ZbPm)RC{I<vh2#C$c_cANhQ>DJJvM^(2;ov<(Q_Hlm8Qlrx{uE&U4pv}QoRt;^+9EM7ShjrI7=P2mKs4B_7yTfm`FMLu6mZmS$v%4v?8'
    'O>)i5tp|fcQAC`WLn1pEJI9b*vGwj@W4!_oK<WgI9z*=|^L1GO{S1HH&Mp}RzQ(ux<ORI>&yp7zskz~7uV%s>px)~K2GxIPXUH)&@LQBe5Wm?F'
    '{=3OuZ!hBr5%cz9mn?@D1{OCA=<XTJi|0k2!92QCad0`wU0tzeB21OT*bDshBvG(Id5+y`lLdzuj2-<Hl|+8g9u^CI+~gGtOSPw)f1hW|yVZoE'
    'SYRE4Ts&is7S)cA(*nhio=IF;6v48-g2qR(f0n+;O!7%u7jCN=PP?N9t%Uw`2b%uv=?8DBx|h7O#|SIO_H8(}jT@Cd=xt@4ngTbfIv_IiPIKXu'
    'NgPh=k4a0QGk;@A4&fqG)E^F8ldu$32b8xWn>Ds9_8~G3z0?k$F?{xTO46>=X*@3NJ*n-3BUSl1@T={A@{BMBU>u(XiOq%5DHCm`P_}#coIK#u'
    'mt*80{Jt$P%#vqvy(s1|z=F;LK|`0x!-t`9$Xc8Fj#NOJw`MtLU#XFEwZWxa6H_4DVC_wX<y8T`g5#;#yKB_Y7+gecY>Veo>%1~+!A-golvc{)'
    '#?S)I3{0E*fYQCl6=)U*YBU@NaDoJHLui-tD_Cf5To)9R6?hK#xHH#UB6px+Bk$>NxlDh<h->D5UvjHZwv6zN!s9Mpusxkz&WkF)Uu|xQ4#*{z'
    '5N~Mz<;l0N`mZ}d7s^Z9^V=k#i+Xeh9a!JF+`T8(hth>TA@S!v_+T;TjtGs@zxwFYIcl<y7+#pCAyW}}iUWkOmpod8|M%pk21xK1qNY<{d>ZlT'
    'B#fomWCmr+9otV>OR>g^F}7^6l9~Dw$+QonGZ{qucLR^|Y!X{o!>0D*LL|D{>O}n;sh#3Q$KeBq!REf<lFvXPO~)v9Is-mxGmYFh$H}ed#Y2O~'
    'k-HU^!N`fNe%;~&G@TXK%N1PuNm>y7r#@Wr)|2}$?Eb%J>kt%1ulr+1O#q*{TL{cmLX5y8#_DxGEbC~>D>;@W#}29q3TvZ$OaB)@Vt{I6m!y67'
    'iE{9cI~bS<z44#)<hS7`<;5MvQmVYhg_pqevR3`C8YpUTPNzlHlZy|-ckfSLogSaO`qqyG7U<<XBSW=*%j9}ChHwuJNmD-1BoN!<B?*p=0zV<='
    'oT~a+`_ayp>)qA{1|at=)!4@GP}~E5S8SCE(`Q3cAa9gh!u>+-$}`KG!@L#$A2B*s^o7+;>2FHyhT!OjjC<?MA#Aus^j<?_>F*Lef7BmMN$YY<'
    'o*T&)h_emOpz7%WY<1f|7JZRwak(_*SI{RuQW!o5?gH1m2fUg9^<OG0Ln1PX&bxvO!`2r1m`Q@pkxRB^#1uq6vnMtgARB`y-3Rg3Qa8q0EB|to'
    'XSe7KXg;d%AFYWiR<crbS7GQz9(E!I#Zu<k;GHP@RPudHhdp+4L_@HcwbYP#E+iuDJ|>GD95>5*i>(U-Gd%IMG~yab>MNLQT$9V@aI7pP@Y3NO'
    'OvDddssV)Fx#4WnhWYjg=+kIVGzP}wc=n9`r0~FfHPWN6W+KHD;0>}_2vtdw?XF<m<$Z2qj0U%*YmP_^3P$+vctnRC;tEf*6zkDn#Z$WWjmP-p'
    'dHP*DETiP;DFqsC@w(+;Kvh~ZiR;SNWQ525m>!*ON{8mL-m<V^$0!jv=AK+6$RcJ}rcgZ{F!423a~T|SegA+yVOJEhg>n$e;HPuZF-HMph;}pa'
    'd`u!yg7uP`<~=RvNqBs&2A3Lu4#7q}biT8+LQe8U*}=N&f?_Kj&$jazQ_lJO6pR$zuwOf^J0)EhyVqm8b7uA|4<di>;Q2q~F5_nZVHde!7vA<<'
    'h-MfGsj9``G)w+xcoMyfZqi&izIWJW|L(a+yk;Mn=`x$xxPJz&U@AVK1yeUsU%B|oF+V9i{-WS=L0tz#&BEql&`$HiINtN`tVS&xcfGCw*sh6m'
    'itapjeWZS^*%6)gV;8<*a0S0B?YBPS(?XZ8Dxc48%XG+GMotVKnZbOM1;zmE?bh%-xu?~>NTeslr828L45lM)ZGTjMJiEw~$9I^Hf}THF#-ra{'
    'Nv`&HVdwrBUz2s0GfVR}+HXTtTd|&LbL%kv?6^m@D5M7Nf&(eu3RhY~`3AfqcwUYCRtU`t)JwbGdU=l0<~%2)=zV90<O+*A0VbsYYm3WikxUSg'
    '^y`=wKWFxUAA=p)hrUdm)zwv1ZXLj$bSYH;-MuW*#<z$eWH#bH`qI^NZaj{+Z9|(1`Wld@j2xm*OE)7f*bs5-cz+5M%7^5J3Lq6*-CHZ1Zt`Qp'
    '01lJzgnGpq#S<wEY)_)8se}lX7Wi{t6G2hWwIsutpX{3<T_AgTFnT?HAVb!aHcIX=3O%;yUP|HLS;FD@8%9>q);I6}mvnAWQv=w=kVCJSPV=;R'
    '8WzbkOr&AwUBrdv4Mf47UTX8Y?4vQFo7h9C)~6H!C>!$vf>Udl6-lrrligxBFDQs1DVb-NJCVVXSj!Fe0tH%O*In-jyK1GGFXU=D1T_bqzMe<J'
    '+=UeA>N|Sy)B;JGp?ooBsTd04JC62Q`ht$L;7t#vW@kbH0WCTG-1XWen=hGF7I|}!<F=3x5*_5posMGD^|k>!LNT#DIYXRuYv7&*t^-w<Gm{_{'
    '^QfYS+n0${zP}@?jq1-?J4O2ZBjfh$j}nQpA-736I!jxs5nnIVj|XSXv?TqL#~WfkoO(kSS(eGgf!Bw`kEK72SsEMt`GmY-xw@7KYoNSo;bCoa'
    'wRzdHl$t==#wuZ|LQ(D}HHaUYsex1U^d7Cw6rs~{;7(@mo-%EgPNfEEPP}TP{x{ahpMHNmww1kL8Zfcd?sATy^O5UuqyD|t<!gz09z@Hs+a*Sd'
    '4mOYFK#lE*iumH2sQXuu!jL9xE*^+|;EI}3AOSz)1?saoBJ7&-Ke{s~?e1>Z?u&oqkIv+`Z*RiBNH={_nLO4hMIPQRG<Zt9nS9e9?|mrjf=3Mz'
    'v1p%Y{q`Xb&NF*jug;t&E<K7MP)Gl3_A+nADk|))Vn>z3rM^D39>s2_^5OwmQ{n(|h`g<vMD$jA-U4~9%LrbO;Z4#fx%Mn8_oE#(C2z4At?YjI'
    'T3+h7t1|KoB-nws*l)*bzA6CFx1N&*T7(JRRqx`pBCv>36fr>F^j=2J(2i_G2*CkzhX%>IAGI-9Hxa%d`v>5dfL-~w<uUTQvJ#FOyhTD0>CIC+'
    '{nv4F6xr3k##8}nTgj}fwh{Cc>Ln-uS7SDUxD$y2Y`4FM8v4Csk3NO!ZL(v8a1oqh^YD>(u11qZyr^qzO7m+Kop1e+zHYcZUyl!&JmZtt(RUjl'
    'rQSLUY2BD?Cb}ehv*qyMG9NPY`j)6^Vi%?}B%o`j{Dl5(zoyG?|F|yU+F305yL#jcN2t_x*w&m}T3h;XYi{TzLO2B}(h`i1dQ3d<v3xpaWaa1y'
    'rw6<$W^<o_QjCxj$<`tD#W?cX5%f1csvgdrZ&5_IFPbmuO?B+tNYo|j3NCJaXXeG^w&JnY<ltXzXUoY}oRn<pL%n(x^jFM~)EDC&jzYPU-buWf'
    'C!}Z)k-7+HjdEG7o;h3z(jq}TeHEn)_lOQ_&U5=@hQZRFE;nRL04w6b438#?O-UXh4N3(|FR%h+(4v27RFVag&w3Z}G2C&Z+F{%*9YV9kxshY~'
    'evZb&utlz8-RlMo_hDqui3S=scQv~r4`S1lZd=R9Y~}>iC+s&a@i$t{Z#HzVTL$|S1d(S*tASPjGDxBo1OUe4_9?hcUd-I&yJ!kD5NZ5x{UkkN'
    ')`d@gBzykb*okwy4YT-d+$8F?g2$4@Ih=PgFV?Ip@Ww(<4?8upYkT^=EFV@&#1o^LafV&{8|#kBBo>j~*tcgEHjQYTW<25%$2~;3n6t?T2h59`'
    'JWGDuw;v&XL}57g#JxuvV@wO%rUBwSu_-?au1&V@3KGU`mOJatyUDOc$Xvtp&ka+o``D)>K*6d%W<-bONnjb4^h*<Y<va(;aHdRL)aL(p&sr$f'
    'n&iw8dO6(P7IL}%>?-VNF_;o%6g7?&DHi)=N+6Zdx8L^onGOCkHnCIu_=kHf7{VKjgLaeAX*WSqhW_m)Uw4z2Xf5SW<)rr#ho+Kd4ZT=nL%+uu'
    '%U^s==hx-sricmmT$q0k4Qylh)x`YD#7)<lh2?0_2AFKiUX<f_cm+2UNBLxVVWT0Ns_x<}pzWE8pknLnbN2onw1SRBFmjX9_cvFw%{Tt+$+ELD'
    '>P9~G=CS!I+Ch9%1J`GZ4l0~Tpn<ZiyjRU!C2QIjOy3U&Ubp`9XBKDwyu|cu`iCh}Nv3L<NHX}0x#<sSb)FIWp@x8DJmwh@exdQAV_1wi>hM^i'
    '?y*ORwiGnUN~}!tZEpg2XbDF>Tg>L<!xZJc0httAxb%SF48rR=30SlgY049yEg?12XSO@-C!Q6mjStaha+}Q7!%!ssSb8BS&qO6;B-QYDX4FIG'
    '8~(_mlIS1d$B&K4Y5Mx1Tg_|2-^S!Mdf@0&k+r0^v7?&u_Qq||piNgX9zTn98_8|ccc6n?9Zy2jXMWPpt`p>k=||YeT7I5>?Y(Zx*kjhW=V5m5'
    '(B7mntb1U7lamiFDwTgFo}k4x+LNZm*4vY9#@5@HV<y&NW2kovl(D)|V2DN|H{I+L9!7hxmpQKf2xXQHk>br1j8H8$bUeCy!@#Da><Ni<?A{-k'
    'bja4<n|{aEj3nW)Wryj|zG6QJn=fAZLD+9LNM2*GVoa9k*{oq=&T&0aIX_|OT8NpRmFHw?Ozp`j?`AdyP;XAQ(`BOB%+UOl0A>7RI=f~TI7RzM'
    'c2{)&&AbG7z(ermHwlIY#=%k5Z8pe*>RJI+yNf}(%(mE`yg~3POjnDZBuCAGc0{6fwdpO2<qkvXLTe7$sYHX~bT^<mA8LVM<Y?qqr0nfYzy-;b'
    '8Cs7^<OSno9y-bdwV~{KQ?Diw<$~Jo2;%1Rd^4n0IB?y{9hW{(*=!=Lv;`}}__sO%Mc7wy)sD|8zDny!$6iccg7@S&x$3yPu{rvTC-K-@Z#(r)'
    '0Bs>PC9sZB_u}Js)o_uyIOs{Gx~wKHnh1T<8Uz$ixuv$2lygawfn@W&bsk?*DLg<J!B#m8j2Mf}%c3gVAzj}nq%~F>)xhn4SD%_)@JZI}eeoq%'
    '0nf*7YKK!1pnlO9t|uIYr^$Zi;6+E=IwPL+kqF@ea}<gyELK4#+H$t)E*k5RWX3{>Yji4cnXPNCy#=^ghxyU$-6bQ|ioYBZsD@lI6N(%M4{Of9'
    'U4ItZeI#LBIQO^$N;o6W)7^6j6bZ~goZ%m86Nr!7m5F;knyhi~uVwZngh1$+Z)*^V5agNp?7JX3D{~6z6JwAg(coj%8biaYkbdIP-zPbb4MNoS'
    '(OpBj^}dG(6@xHzHENOchdS2Ko^;Cn5D9BH;67{|`Q7$i_U)!O$mWs8cZftIwfPz6-q8~~8$UXK<h}rvVl^p=yCeM?!;AErV~ePLQt?<Ef!+|6'
    '#a7r8o5j)GO%YmbjrKS#VYO(q7R5}qMly3;LM&KXI8<ZEG`CA(w3B`!Zc8m8NBCAW`t{q+W0B!Z;j?EvM`K{y5IRjkB88K1q0>+z3Sgh}rO(%!'
    '5`#o>kP_s8i24=Ir<oEqp}%Dd2E-fe%ijY0sGV-W2UrVLN1d}IQ}YJ=dhyV)%p_KgU<OGLwCZ7q&7&X73_<a)rQC=sIVTQ(<Q}Zb{T4Qwys+#h'
    'Nj*x#W~O>&0wGy<Jv=@W(O+V#<2qbQfJ~ycb13s@(h)V49>+^b>jC1-BnIZh#1njQVk5wlFglXa!bEJaFoQn2SUpEO!&I40vXY}3h@kR@0wZQ>'
    'e*8_)6fueT+Yig0r>v8QcSX(^{=|bczV=Q2-|_wD1RK(kKbIlRU>LBnbA9uv#y#BRO?d^=yt1YjF8`OA41MWQs;hJ~9OD%oen0%DrFn>NMRJ^S'
    'HK`&1i_R3T<qlp>AYIGZ^$k3J<)nD%NCGq-$?(29Ubgu(YU%;!j~Jtt<1OfuVyc+Gt$C5F$Ml;E675QKZnjLv-;7tOCx08ZCVz`}(VqX!EW||z'
    'k=$*Zh|Clpb9aG*s{n@!E~=o_e)OtxE-YP7R62`w?+1YBk5ry&T*<k#2B$AxqbkxEqX?d`fwlY@38rQ(oeTF+quRViK%>=KG!bxKL#+~2gDF2_'
    'Nh{GbuowSAxs!ALp~#ZD^*LvaAt|Gyn7etu%!f6WznCt3gI%GMsTd!;-Fz;3#a!zcO=y&B<e_#;c@pK<A0r0zDgWZGn9U0k%_<!il`?>^9w60O'
    '7WeQEcoFaglh|eaptD?2U@m+cRcA`ThNKKru}ad`C@+I*^a$AoT_3Iw6PX<*Dl??)A0aB+&!*ecG#m_Gh%0=g`;Lz9E4xJ(hqcz97j0H?*(RK&'
    '&21B;7onsM670!{bVp-*d|M)HI6NC2zyUGZh+V!Y5r&J+!}qWU6M%z<6o^A0s0JOOQIBi~M`*o1+YA(AlVK#DscEXkxTas2);Ta(sjU?FhEa5F'
    'YjSuh&Vm$zKXm?Ip+SJqJPjf~m72p7fqFYa&ppR6w7^=6r<U3rb$^Xn+PR=%jqqWsXejw?F3Mxf@7KVhigyO94nv^Xj$Ozzn(R@+0v*2Z5>YD&'
    '!<#j_ivfsPpJo`;mHtrzX;0ki0OUBKWt{#MS>Xc!IFQBgrSN6Lzf&gs-zQw%D_?BejzcvW`!I7<U>U*CHY%_$G@L(Bys{-Quse!j<RGpo2CpX`'
    'e~ng52VCnoy7)F5x41n%HIg7E2I!+R0?#bsgr%jWGX3^@1+e(q0ORXD&{(?^uSq-<F7N=EbfXYHK1A1_rk&91+Suj=`oa_Ywws*3{=<(<B0kdG'
    'qFzK%#8uBut0@f`Tcnb^B~m7v%?<+(0am@kG$YMdzy<1i{ruZz2i-jf`Nf(PdP5@?S-)S4A{rEv!;#w7B5*b9RlJOkgHVd%IQ5vV?A9YNe9pcH'
    '(<yqB)%qcC3=Owk89u@UH4fEWwxw9gKqDiVOxbF@zn4JOX+>3rOEhx@$Tj<BqBnY;^4Mm6*c$adW`0yAU8Dnk9jb!f3zecxu>@WCn2Q|y=`Xo7'
    '3`aUZEfh}4Ctm7Gq3c7GLI){^!ja=DA$)8bbV7}DT(8gu3`V_VK`n;8$(5BwY`Q_<BnQSav_#Za0d$@`3kGhO$aJ=ijo+gZsTL0qIHmMCtK}df'
    'epEr8-p-5d(9|V^N)afNY&>+IB?mBIH1J&@0YyzI6HXT3lQ<J>l_3)ikT05mRcawRM*P>h?yXmA?#EBO@n`Gcw7^8C6Etj;7lD|9+tL%+Zwjd`'
    '4`?lTag&mft-kAeKp9-?0uABF97|s~+8};$-#y}u>cTcfVPpQ`L)3(UP$5(ll2TwlPV@AMio4MILioc|_#N1_)eCwNxBJb_d~(%~yWQTbMP1d_'
    'EEdbT|5Wx{hvd%17+RIJW(7D%oA&>K+E7Z`w1%Ec?h~?`@VvNsh%5e7wP5m`ooqL?bdp}WP#_SWbHt{$n!+hPH}{*v7aDGY6U_EbI2t?9`^%gO'
    'CA9>SqahPe^|%U+*DxuJn3ZLy*fbbO1i4MDZKFU}YknGa1p$(jd_jGaEXX}`COeu1V|_MCRnMNhG9D!hQg-w@*HS+&fDeL(NBs$e2(|@=DDEb;'
    'L__^V6l?X=LVYi<5Z;O)Uyxd$BoLz8sfD8bgWW>D#5}|um(c`4le$Wb2y3#vaSd2wglnuTtFBv=QZ_?NT%)C5Bi1IV<MFjiJ%fKIL>x4kcs1;x'
    'XG@D0d}A<r>g{EiF1fWk!bo0cq$kL<hgnk+X?*DDR)$hEb3OM-*ZRR#lU#@W<lXzXXCKbqzd6e}<)OY|pF1uj@}W=zlUPmRAlwUHyK*j`K3+I!'
    'KsZ&US@3Qm<_-jQ$My8#Z20}pzg)OVkB8nMjJtXO_l$2u#9TpQRT)^W-!vb%+3JE5*AU&pd_5&9r6hZ+4gMet0BhNhJ^Cjt0lsS|Z32HAmKZov'
    'jIvPq$p(fJKaBz~9Q{3cdw%*V;mnN*{2wW|=re{iGx4Eizp>yR7>7SB=fGk`=>n{&#~`t@A9bXV0+-v5R^lK^Gy+ZZPUp&!UK04rYIZ%JtzR&C'
    'gMOTN)qCrQP>IMr^a5C6EukCM!Hdg_W}75bYW(8DTzr^M3041c7^n%6XmKQ^LlxwrXH9ObT5{3L4b^nBTHRJzv<L`|Sn(xL<hrR%%j@lKiOivF'
    'Txx`}W_9EpcZ{`;sgqET26G*>;zbi*nd`4-p8#<sdQUCLTO=)eG~j=_f}L_>AJah*F0{k0*k7xr66Rrg^rqqorfTQOw-}HE{fGBUsvjQFj~n>@'
    '+Qi&%wy1hG%k=k3#nV`>$XF-U%y2B+hB11m`B`HWtVX+sfWuy|R&x?D*G@gvdpM9?aE3!!kD6`-r5rZZ*3clvDj*07!ZYNR8DGxREEZMX-ftU-'
    '4Oo(DHr<gw5@9F0+&$0F+5Dj~h-XTr3x@trM!sWq*aS`i5VA%x(+#0|J$8c@2t%K+vzUZGL7)8)f|jc#dJH>sPU~(8uSjwF8{?}4P{5*OwS~XY'
    'E6hR~M98@$YsJtCw1?OP_$**?iN8k*Vv1RDsuX|s0FkDl(%%b<g~8<alT2ozsJ#F1{)amLw4#!GGQ+;-fmw%uT%nr`9nzGjXkKoQe%iv>676kE'
    '^5|2QywA2WzHqM2cEyZjn)$D=Awlr1H?}S{g=OuHZqN-ha!q?s$I~+|Xovo&AYaAD$NdP&#5fDY%&$Et-YJoj+dXKPn>TL3O*vl^*T{cx-lst7'
    '8Z%uA0JOors?EU46DRI50^|22ZgSRab7l-AYBpzkjm{8I=N!cHfV{v!#UtA5dR%zw=-h79&a~UisAu|YdMx{71IiS}8urYi;BaR|L~XICvsi1b'
    ')MF+BPO2(XUXK|mIBfH|XCC&Dox**OL{%V)Y+e6Obk*87xRt|O_uMmVXXB!%k)f_-_KD1LBmMW$eKZY4>OILKwYi#SmHJdGRx}WTO&-=-XyaND'
    'L`P%~BK#o(RWej(I@JWFux=j5wmY9H3&OY{$JuYlbs5<6j)vpW?KpDoI%#kDX`L5$*PN4{{>-a?b-ehq#A7X13re0sF+r|Y^ZDvNG3JGxkjsgg'
    'nX?pkx`7@yPhoB%ys*``?qO}D+X)@22vpOMAef;oQMtLwWqV=NBPFB5#AlR^(dMELY(0Cm-1L6h>M4Nq9OK}WpC<4Qx!FaP&(?#2DB$4vV6;-R'
    'VrVyOP!<|K6xHCXF2w0K#Y$?CQu_&a{k<Tr9cAEZGvEy-k-a58<RSJyrZt7D2vw}{zb8F7-(A94C6?bAp1;E=8k|OwjTq1}hD<NJ$vyg`$Sx)r'
    '7t!7$EkMxv@2G|2qFBzZNL9sZLiBM)t7M}?ZdyVBk&T`&*73#Bqz&f1f<e)BMf^wD#uGkt3>A&;Uw24QbWrfW-SADo|Kv-=y%eRp<NWtA113f9'
    'CO@4IzrK6_LyoB^KPLYN2(CS}`lEs*L*JVY#}s!74%-W9Emo;83sVnn$6|F?R=X?so_4TXV#Um<9NU^*dlVz0x6wM&3}z1JcME1L5>jpHnT)h7'
    'rp1~a-&6;1RX4Chy1jg|J+Qn>wD_uNo%$@7ci53#HvFD#RvfLyj3+E(PFnMUw{sm+=*%jov+CB0P6)qv9pE0X^SNZc9tfl|ZB^wMA!@)nuLace'
    '?edMuYjxYZEguBUJ9#d8J<(H(HAY6_JN7HF&{tk$If<SYVq@cseV^T)$oxG8vvE&}_EJ{^(6nos0J@i^$49Xw!`XWUW~jF3j+=!d;j#9<)rM#a'
    '7tpGX`5_GGYHO*M%Nv21p@W<wN}PZh0&p3*-}r)wrGpgV_2KUVqFBIxS;2cnl6O&(YWTy?0?BqFtBrQWr7<E^#*gqni_%KGYvxTiwNl<NlV!bI'
    'U#skabRPqLE^?qRVrCEcRr4<fedObp@`5<9l%8#2NF0%9&=eAkynOju<<mfpY;K-yrYbuwk%nt2z$c4!H_seSAnAAF{|yif@GNx1ZFsZ1Mi5t>'
    'nKb|fqq^My7z4)Ds#-glu)R4xcLU6qw%ci1n^`ky--r%$MejaYRPHVlca0kzTqTlpC}8NV;3lS>&1HvNBLMWuVxeBUHQuO1&FYO+7M){J$}J+c'
    'o2^SW8_w{KO7#Ja)RUhImzYOHnWveBTlz>Ko+a2EC@+#2($0vUYi^U$#q1{aiP3zrs8BX0Bk~iPa9C0d?~UE}@Z&U><4a>ETGT*{C2Md&X-8tg'
    'T)jQp)2{Vgc$!*y+k659(G5^KF<k{K&JG7{XSAyx3?HtKxpsryE>9rtwOhhIr`mZ}tI{^bvZ7!Bo63+V!`*iE4)q%6MvG$TQTd53y0tnz><@(p'
    'L{I@S=%H)dwuj*lJ6#TGozoko(Jx9F;FTP72<QYbgTurg2%M|bb-P{V99R*yae3_>rPF_o71kmhjcqRDHZ+n_W@PVSacVEP!`&ttfJI!m8;DIY'
    '>h^RDtzO(bHP4CrkMq{%&cz7Z`J}J;Xq;&$kD}j5QfG+I{aw6ujUKZ~5JQ+f`V=owJFHUbCxWl)YQ4TY2+?i35|-kT5`JhE-hG~|Hm$MpM57e$'
    'JKeD(srV0M=S1J%K=e^~4JH$?Bl+{uh1tP&2K5TUmAG*eKx%Vz><>iaW=nwUCSkNN+i{=TXU45of?gEUJG6cXbZ5*Xzv0{x{C0GcWqAkqrAQ=x'
    '$ni~4-7w$%9eHcxh*$wwK0s>!hUbP2z$%aBrb^1=u2+>?5tF5zVa~-Jl|4@~T(o7NXGyYA_iko{g19Ht|D?02pPU}eG;tFmvJh@<1hDWYe-VKp'
    'J998<z^|w7D+ecFxh$2v;F!HcmJFf)BWp8lZ;E*J`c5c_xk{1e8$40YY07+5ZL>yUtQ8%{XP(BV?kfbWL<!em*Yn)USk0mQ#gu+MHtXay*(Ks+'
    'OxS1>zL(*`EzS$Q_20XuU_AqWJr<Uy77T}c9gd~92uUT^QW|KeA6Y;gTP28w;JQGsYr65TynK`!Z?w@y7w8Q9NI302lQc}wL%W2rB_K>_wPkI!'
    'Vus=^29lif#5Y+Id>pZY(l&jrMqN7;8rwb5HqUH)KXMY5Q|cdNd}54-xy9(te0O|YyDi@M09(A#9&61;S$$nlfr?YM9WHZb_zj?VAH(nojN0D<'
    'fH{*b)O<~yi%8eqe|#(`gd)ZI5jXkoP>Ops+x}suUA$@>X1hgMuoCOVd|8|}g6fB#jLgA4Fi{^HU<5WO72<8yM%bFO;SqXfOU=YA{}Ex#tdayp'
    'sx)W&#n1Yu6)J6YjX_<jRy!VsugT8bTjIVXMJyjR)6=1f3~msMcAGFLZrg$#0fA+#P*9%($RDXm_7<gLv`=75AEu_c645%6*OaP}(Ut_adG+--'
    '-*l7H({JqKz;(zT%7|pxOpE0A2Dsf!j7MdI1&vv{G=$5@8%q)YX|QhHJ4xpnMtcgig{~j#I$%vQ>f4-YIR?qiE&;_&iPft6TXmD+n>T0Y7iZsx'
    'Cb1;uYbm$#QRUfK6F>hh(VK<SLu58uZeun@`Kj%d;OOWlDiWrcB%|Z-X6lCx3BBa$=KJbCGl8y9{jnIkt!}^EtVLYpPABe)<%y001a7ov4W7>~'
    '7F)zPnYf?ZaptL|#k$%PL@Mc|F+EnX)QBF?an{!8?6Vcgrf;C)0<&e0(V2Qod);xF88&4CE)J_kR66$BH=L4};1TbsXZ2(TG1rqNJ9gV$)^oH='
    'Gx(%6aw}3{p<d97%+e5{+j{3xtk{YZp4JhH={^TMNI3<n)NW?XP{|MGU7I9@)TBx-<Q9pAfErmpq%xUN|MbP{vG{VdgcJ4>p+tr-yM52WG6|Ro'
    'Gnu*)<CAOisGjThJtw3_N4OWS?JAm%LS#9yhwZQ!e0FC=fYT(l(XtxzzXq9&4ykA8&w8Ztnh3Z1r78F>D9{0mt31=d*j*4aDi!!IV)tscvu?2E'
    '`7E0&)G&N_?f%fgWAJ#<wR36-USS`S)qE<puke@{X9`2cUSydzN1{?=&8hKt1dniK3h98BUM=CDf7x?##qJsve{6L52W}o?5ez97;+2@oHioaW'
    'kbXH^a#%HP(Zw`@^KyB;y&0UIoV;#1Deg#_>SO(`5e4ulHA`t-%6aru_9i46rH1K%EYfNfPhTL4A)6I!bBrS&tNcuLaf12Zxt~`w<w;{Guxwva'
    'G}}cFMNOY_N|$%iP0Yq!M29;b$c~TtoI)aUB;|>n)dUmpq!-Ciaz5gY0fkj?`Ws}5aUV`$qUkTK6|$^{pp@>tX4WmDJn;^wJW=WC_&_f9TEN>x'
    'b=)UV#M_%AVq(gZV`CnF(MZsHG=__}NFwSBbQIQ7j9)R}uri*%FsAdm(m7N**B>NuV_!}LmG&p7WIwT$g2x|4$@ZGU7{7Frb3ouH4lnr;N0^dS'
    ';^w%>IV7j7^BgVKbd4f)Aw0>zY0Dy&x9=`4&OV^TLagKZ$@{b6hqI3t&acin)Hw$}=hsg$8tiAiW>Y~T8WGkS(0{v*{utQ5vIy_&A_))tn8C4H'
    'c`e3Z9K|DT>;enN!vVKUMmaP#jiBH7W;4X*;2}Y<5A*7{2p<u4DG2L<27i8V6_Fh^MO;y4a~%3ebSx&UiYHw92-gmoDl`ct*)r+&pd}DFlFiuw'
    '-jP`Bv$H?vIYiCGbThj;NWOIb?k7`3>Pugs<TBE>%8~4@+ejin*G6Mj*NIWt0psA1p{cT^ut)tDC=}GrS3s9ZNXS>#kWAQIlbYC%xfGS`i0vXZ'
    '&2f7%kaA3u4_vNTRq(1}DKl30-Gmu?WGYNLT1%85w-zCR?4ESZAQ3FvE~?-`p?m0!RuYjxc-ufn%#s5N{DP7LYT1wphPT!PSfpRbZQqp}Adx@('
    '@Q;oru7Q0d2B4P9h=4KR*Kluh7h&y1xc6<RMhc_NzOX{<mB+Rr6cWHQ9JM9T!{(vMezdvA276CEo^RI~s=8$!;z@$KZUq}+@VURuSm_NtxG_Pf'
    '%#3>Y+>k{;>IHl8rpT?>A|Vp{B|4IqLQ4)v8P0`4rvj_a?C?`JD#OVG{JF%N6FYJ00m>g-SP|QBrd_ro!LR(8HC&1PfZ1tFa<0_*_Gp@|#MN7o'
    'J$*Xz1d1O^>$D`EnVQrAzV3&g9Y0T<*059*)LwjMgNXJupW4{P{Bxb2_el_1q2*8TH~w>dkG)s$U^P^&<(E4mJEaf>F=~+Zc>cu7nKCN(@%=cX'
    's-P!3G@I$|@nt?tGO$19<w4RDSc9Y@j`blCuo1Z<HA%#xmH!Db8AQfBMkn1iQHo)y+-{JAwA(*E@yOk8w{(%YHgm?7;!x%*%CNx{Q#c_@QpXL?'
    'Ik!;TDiwL-Svq*TVDqk`Jq}HuRC0jPjuK1hIuG9Qh=|Gvj<Gb8)D~P$pL29tDlRLVFIA;`h0kyUpn^#!Tx&dk5o;I8w2C`&0~ZZF?^hJ^jWT?Z'
    'Cmr|LX(Y}&<RwM%PW}P^U#dSI{$fZ#&nVtQSYupfV5F;dzL{q0NvG<qHP^NU$f}3Rcyn^ZM4!f^bV3M#=DQA2LBu6y9S3d({s>(AyqKNBHSF=8'
    'tA|@fLYjF(v`wdXEesjSWcz$ol{Fu8lh>*#yr8;5VJdG&AOx`v9XS<sRnb&|sIxSMM#P;8H}~urfh}5V*qq#)UsqV};A?}sKeNI=qYLBurd*U8'
    'zywZg3~%`QCPCas>36>qgDpI;4f2*kLLD2QU!SwS@`9c<<t^2@Jp3#rzc#x>9sc+RzV~V>CXI2&`rgS_QI^$E#uuYlYQfeT^&)>lkc7IS9EPW`'
    'qU^|iUZ5h6%1IsqwOfMDV6mrnVi_uaCN0#}Qk&#uu|PHW;zlIrhlC=rs^|%3I;ZDFM6%``Q$Q-G7L&>fa5V%ep}V8g#mL2JIVDpN1TT9M+0+{6'
    'KwJMsNTxj@3~)t^%*|rr0igSye2{Eo1DjjI>DElqOC8Pm%7COPHU@FVY?AjSjKkzv1TmB=OFbtX#=`v`Nwsl2Lyo#;&5g>%di%hpKPm50ZF+Xc'
    'nkQ}#-BB~%7WxUu0(S)0N|9yuW15-IWWNsAe6_1w8oVLw`V4LBaELGQGN(05up%-rGS;9jF%l+4Lce#jO!1G0MgH>s-9`TKV))_W?EAoBrs|1!'
    'fOi1toBY>zAK(4sr?dR~vtQr6Is4clvbz%tBJL=o13N;5gGMB>Q60u9^~V4BY4|b!kDovM`0;%B<}82n^ZU2&evpauU!HvXs{dNTBFtx#+1BJ}'
    '2SV=HlKn)hiG5@}1))3E9_F8-WV>l+baM1<adb8Q{Pl}R7$BV3yZ68T{APIZ?&tUUyYD-+0MF6@02ojpT58;h9aQ7E2<xz$cN9lqK1<1MJtvuH'
    'o@Yq#uI4+kLq!wArA;2fmdx9S@VE83r_bf5a>C(wz0Y)V4QWVpCL#NzDpQu9-qqIPJXY;A7y<+Mu0^`F)j4NdG|q#7uP?c`KH;r``$ltFBgij-'
    '@+p<aYAgzc1jwRu;ReQ7F6+g7-pOa*1>A_)k642o;GPWPfJ@Jj2zU}QRF{SiJRzEGEz$p-tkx{t8QN?806V=j>p-Am&_F^5EmHu>?#i?l=xCF2'
    '!^~OIcle7qN0_-7pJO6T8XE=auW=gUeNwL~1#DC0@^dLi&@HmmC@HDu6>Um<dO(_I=hi`SRA7WjIW4{X&g{T_$U)Tng>-JWs@$TbRKC4I={1lR'
    'SOp$D{;8vXaQ$nkPb*)oZMzKJHl%S-<^`hhM-6*v>L5Ej4h;RN@u2o}_w(9`t-W{(<!~c_XMrM}>#mmPPNAmW;&wW-bW4lnbA!2ifSEkI@DW9Y'
    '%UxWVfsceRME0BvN7p-JxZ`Ndto3~ivlc#mbCvV`v0>cANpw%Fo#tc`Fr$5`EWS96Mf<3YsVi&^1jBL@H}ehj*4}LQ>cN8%!dw>W2>&8S*tQ~B'
    '2!UX}2EKL`HKkv76HbTWEruRL3E*HHBx?~kz1jBms7_&+%sC+jjZJWq8HR&Fx4+=XimNRLS8sV1zw}+BzP)FSJPeF4xqbJjp5BB&Jd8Fxd5N(K'
    's+3xr!vQuDj*>IOwNez6or+fon02$OifbeT<<o~>J`R64%Rip|^cLR79SS8kHbIQJuRcA1x0I%BKj15eL3?phE;^NiEFF@HpG2&UlmFWwd7V5<'
    'PETIEOrAYUUI^120ZL>)_9Y*!H6EzOPYf58`SGI6h5?DV4&}@Ebk5c+=dJsf(M-Hl6gJAfBTP-0NS!ASv);RW4MZ9M-HV{DlD{kH&iw{XSkAna'
    '@XI%|wX5SC(In_O6rGwJ93&n^$FqXIzc_jM`lM^(dYFJfuHDO*ulMB^kR$}!(@WA4NH|hm#M*OOqEFnLZK@OfiqP)+*-(7u!PAOI`HDx~S>9e3'
    '_mLiJJ(lAv4CLf|Ezr5Uc0Z!-aONe%Tcxs5@Xa@B(VX&lz*Wj`XO|dDl(WY$D&u>L!d$^Nrhu+5u7|OdMbG^@!X_Cpn(kVk@dzFcSf1vrA3C|t'
    '#O64tWihp7Bxhc2Z_v@^w#tD3LLGa~JYueoTOiu@Ko<k;|5bDI@=0^Kx2`e0i1`&9mr=F|c=M%+U?qavYD)&q<VeTfWhLdUU{4IJgBlk<+gGda'
    'ohuX>yO$k>qw^ZZhjI)6V~xNkJT{&OGcOE~!lu~I4xZy&i>qju6Yg7nDZC*C1gSmq5u48tN3F<>9sV{Eg=NmUDHUPEC=fJo-m=SiQRVlm&26<V'
    'C>1sN-t-Tbo)}M7$wi&BFuDqcj!JGY8N#z?OaMRXh=&%u*mWuu5bDM-+8z;vdB_2NOE&58^QP)Fvowf_F(N+?iXLf%VtXFtt{nu3lgWICp8So`'
    'B1MQS4ZiArwJjFNmiQ5z{f|%e*LEh0by8d+W05rk+6!t?^b)G3F^RPwQ8tuWkBrFLNwWi0MD+W)`ffEL%H?1jq5OE>S_)j;;d5^V1PRl5HA4uA'
    'MAS2vvNb)Wbb(nL(e|`hm+8@Ih5-nk50aDKi*Mq7VEzb5KDj&32g=fd%Lk-2OpqpnB?5)0Q-(f0kEPL@&=MpHluW={YwyipsWw4USK?M@Pt8oJ'
    '-1?)gO)+!T-?MfO&I=pOaT;q2t}nKEhRW}h7LT@Jtq~L)#NL(<D)?2J8qyXi62P3-h+tDg-H{Rof7|O}zc^E2`^kAv$_E!Cp^G&1mS51%k0$Mt'
    'N{2M&7AC+!H#yC+{TrX$4tPBST2a3#fbHjJ&*VO0(4R*ev6l0L6?2LyHFe>vB$h{TLX-;?y8-0LI%q+P2P7`;?-T7c$5v4B)V-h~CPO>3N2kbi'
    '*$ngKh9lB%(jbV=JQOvy#EVyqRrLUG9{>kxE=}odavK+V`&Q;Q&_&?osN?2mD%-rZE*FU10g>#eHI)I=eT+x6@A4F+pC8eT$?pqrJ5X*@8*SCV'
    'Dtc&ORE{pWF{X{NJ-Q7+JR(7WT)1E2`^hK~1)1AbzJy0WyOCR=e$?n{7-lhEAQiq_$iB#*C%qGqgiJV>pSOYNYBjQQQDdJWLBf5nu6#bbEz_ao'
    ';er!Fmob-jOFe@PAr{C;|LO|X>a59O_*XA6yIU>QD*)Mwf-%v_jksZ7+CO(b1rZa*;I*be4I}9=YPk?gt#X*oJ^sYTEfpFJ6<oY_wr$}<t=k?g'
    'kjC6lMpPrHBGYzY3z1^2Pm;_EWZB*1$E=o~)}CU-a*Jj?Q0680fu^7G9EN0*i$?D>Z3bH)KAO$5sE5Oo3Q;zJ&<aCLF37SC=V1fdw|c;ZfZZc6'
    'L!M7P5x6#9!#Y8Tj1|e_;50DZBy=2lz2p7w(LX8H&3eWZDXkwU-O=X*G4g{CP4lrA+E&IdO|>PGKGZdkGLq=}p^>yh2F7*dGu<0_h4CL{OlAQI'
    'UQg*XbTYsVBcM~Did*3Bi%==NN4q8ck-J0!wR{y%1Q~7+v+yxp`L>hFA5#IB@v0m)1DC>(d>Zlo6<*D2O=>_HHl`G&AxhRk#{%5jaHDO`1_lb_'
    'd+JXLzFHE2!U`N28o|3{K$S8WDIzGGqA+)`25kzy+s@_{usO3eQ>|E@r=uafkf%KuyWDJ6n|yYKZy=w8eh3Lr5VJGnp!Se&Smr?#F24p%2HKb`'
    '3$;|W4vCoJ;fQo%YmR!V3(7PtDh_D{9}t<}&fq3ZkL^N^0r0Uy6I>1-!b>vl==j`+BIu|yUxRTN4d9FOQHEvYe+8#Fd=mAif-tJncmf5}PB%`N'
    'x`~T5kf>}?Qb`71%WHY+sl%8)wgbOr8ANn@+xaN7<*R#E3vhCB0fW#8*OM~3A9(A0Qq)QAI4UgtN`p$7X~$%1Klu@dARUA}E0WQdQcv;Qfx`I}'
    'X%Fnnqs#ba=Yr=E;|syqF*lnoEd~^o-zsTvSjEFGY(7SJUy2z6;tyhmMTo5olZ`BU$YN|Iofl}P8-=@FyTFqFH<r@9mbQAsN)+><84zt=3j_~}'
    ';yc|f))l;#q_-oB`=9@LK74bL|8oBQ@ZzjP!o}77BO3T@wmC*Bmtaw-j=Im6#ZruN;dz|zsvAFR0gK;O_g_8!UuxQYF#'
)

FULLFEATURE_ORIGINAL_SHA256 = '63d5d1bfffd2881a0a4a6aa47a3474ea0483fff0f8d1ed729784fe91227a72c9'

FULLFEATURE_NODE_SHA256 = {'SCHEMA': '9b68fa5c5e3afbd6e249f0b2e2d4f6082b6334c8961a1457a7525c6c7478a6fe', 'AUTHORITY_SCHEMA': 'c48f888afe8fe552a1303492c33db760b004c472068ea7162300bf46a2e20053', 'INFERENCE_SCHEMA': 'a77220b4dabd76c3cdc21dc8d3ab6d04651092a5f738cda9749be0bb41fe8003', 'BUNDLE_SCHEMA': '0a5b7a967e5267bd828f1f068e9f5075ce056054b5f8f599ea077290f482741f', 'ACTIVE_OBJECTIVE_SOURCE': 'c943f4407efcd87414a674f987d620fbfbe6f298300a0a902d031d35a5ac49b0', 'RECIPE': 'b1b652bbf6b4d00b567f927f3faf71af784d32031b25b47b7ff87421a1b04d13', 'STATIC_KEYS': '49eeb9afdf53c3abe8105027bf43614e2507cb20886c082d22d48367bc9c7451', 'PAYLOAD_KEYS': '3bf4ebd170f1d1b843c36d76dd0a40a204d63e870f41af507796d6d3d9d802b1', 'INFERENCE_KEYS': '72278f37af1082c91f9a7186974425c0cba372e9dbfe205dde0a92f807d2a5d2', 'authority': '47b945ce7e376e4de369f25a9a0ef0627509919fd99e051122d9c41f3e46e3d0', 'prepare_native': '5ce84e1a0284d416792122410caeddebb14a3e7c4b48e16d8e6b2695d0856b7e', 'require_no_training': '25e8a7a041a2778641802a45f437a7202c203cee6e46795cea8748944f15564d', 'parameter_roles': '17221a0f94371dd0123fe44951e86d3a97dc2449d902bac8be0556b998469aee', 'check_mu_train_provenance': 'a7c29161a365bd30f8c153fc5a2afe34e2452183ed3b9c4ebb76ab37820d701c', 'own_residual': '6835a6943d7a78da7edb43882885308882f9121b885973a7dc84675e20faa1ca', 'fullfeature_raw_features': '337adfab3ecc63f101bab911497fcb8c9aacf2ea45bb611343808ef14765ce9c', 'residual_facts': '7e567c86831e6892bf9d868c0febd083b60f82a896530d3cfb3a930059973477', 'own_A': '7daa0ed42a9eb402a4a2a1c8978dd300fa3451c283f5eeea5d937a345cdf2992', 'fresh': '178fdb25ff58c1d54b62b3faf5475633d4387739fbf338f319f80b5db23723a1', 'identity': '60308129944edf96d63e9956147c20b7d7f28d19bfae3c1386b66e6ff1f92e1d', 'payload': '67a9119a750659545d4630b5cbd80f1a897bab800551f2f499592dd9f31def85', 'check_optimizer': '8525a2ba87d989912eab7f9731724033cfcd747d360a2257f3e6daa46cf12eb5', 'check_payload': '4dcff8e89f627ddfc48ac1dc0df75a9c10aa65ebde020e4acd64f9fb480dc48c', 'integrity': '64877044d20931db9bf8802704a6f70d6dab322b071dad189c26c0ec3451c633', 'release': '73b095a67686e93c4b9717805a59c88a9bafe291b832f7825a88e8bdbf8368bc', 'raw_features': '0b877d75f371d1897721897b3f212d8fe68aef2613bd1e6edbee9e53703ab9f0', 'ranking_gallery': '80a6ab2cf778da8b8f6c78ccb183cd877085c4b62d1ac83655aa6f86ba69e3bd', 'authenticate_active_objective': 'df179a7505fce93a4fbfdf0ffea209b2b4a483cfb6d4bc8dc56d30d4309de3d2', 'cpu_gradients': '35ab7c8b330c3f612ec74e2ba3ff20449db5811457e102babaf4ca316e6de600', 'inference_members': '0afc5b5818e904a865fd65a949f30efbce5ea4cad258c2b3e070c8ea97c253f0', 'clone_inference_provenance': '7f0f106547748083ee34228b7929f295aca9c20370ba1739e322fdbe19680462', 'inference_readout_tree': 'ce7d9ab4fbf724759b041275023d9460dd89c5532199b2ff6358a34c4fb134cd', 'load_inference': '8cefc90f00ecde080e4cc29c43ab8cf6fc27381ca974157b5ff7edb0b5993f7c', 'inference_outputs': 'e02047621095dd855ef0e54bb78d84b346e0770bae1dae881561550467b6ca26', 'qualify_bundle': '2777b4de8cb4bab978539865ea66ba95c64146557088c05fd80e7477684af9dd', 'update': 'a0c0b8ca6fad2dbb8f73ea88046c479cfd68865c0fdc31f05c4e52ecce50673b', 'check_steps': '93afd0142e2b174308612880e6124dfd7288328ef86a7278464c63b68a12a90b', 'tamper_witness': 'f88c1e3ca60a0a70a063f7d6df9e4cc37ecb1a0942ce2165eb87d7008fc5471a', 'inference_witness': '05e85e693991374af1071b726d1474af28c2655e12f2e9e312ba1b8ff3b81d21', 'cpu_witnesses': 'f6aef2f4b6779a37dffea20fcebe6ee5366bebeb67d30ff39dda5e2b84b5c6c4', 'gpu_run': '26fe6c9392f35661f028dce1188433e0298247250f6f3808198cd9138298ed72', 'check_cpu_gradient': '52680adac8d04761cc5b5e8020b63d7a0760043c2a4d0f9c43ef2c2d68c1b47b', 'check_terminal_record': 'a893755494ade5719a40fe9d4beac08b8a095173828b2d269b62e8244367f307', 'run': '83f8b97b6fca293f14bc24e9f5c6eef4e69b780778098492e1278b279273661d'}

FULLFEATURE_BASE_AST_SHA256 = '4fbafe0781aa9a993b1cfc7413eafad8a30fd247b44b247bb4256f4436c1238a'

FULLFEATURE_TEST_ORIGINAL_NODES = (
    'c-rke{cqd0w*QKd_k*NNb?x-y?&iV3eA{#@?v@ozwuiwn1lgh;-Pn>R$w_nl{J+2RK}sYg*-qPSz{T6p*cJ~D$;0!_Lv2sqWNDd4v+`OLWzj3)'
    'Z&-?am88)!%*DUNtSrKHl$Qzoo=3|<gi#uYX_n5DG$}<mTW`X3o@Mis^OIQ2oiD)P6=FH}&u>x({+r#3*_|_V@L4ZfC!R}Bbkwsfk6oX^m(eDj'
    '-3Csmy6^)}HW%3@pNYVkFQY{PAl}yHNA9`f&XZ5&CKv85aJDW%7*8dNX_Vr8u2xxA-i|&Hy*$5%kG%?{oZSM<GAT+gkJ5$k24{iux7WS|Sc%nC'
    '<T!yiPwqwD1E9MkU4*jXHd%X!Fkn=&R}zb~gg#q$CCb|@hS|yP{M+NT`him~N-fxp@)fRThTEJiU4#(=Q4Id}_x|ev{Eo$a!i0dGKmgdR2r!F0'
    'OyLuWF?L7p#AqpQqct~;2WQ^~&fs-_f)MFuCGup3TP+DmpU|R~itEGfXRlsEEB+e0^5<m73+An*6&JAUtEgV_)tetKej4FaZlgk<^x<_fmEeiu'
    'l@@9dsgv>%p3zrp-g3eLjuT_YZ3<&Xu5;?Ve(B7!+)14zRdap!{>O`}9nNh7+mtifEfBH9S-8)u$Gv|QSxQq@WO1jA)94*GnsIu9Cm*h}WiorL'
    'DaF4%q7Sl0wl0%Z@^9EX$!tg)cn;DJ@-$tFuz+og==oXy<+puko9EfTMH=2GMUtgjzn3pz%iZ5M(GsVXiwFt*4rLn=W+iMtp+&mERsw3+(KMzZ'
    'dx-cUS}wJRXY>%1z)TdNEJSn{=J4e%nPQKAe=lRrvekMCOc{qoDb~<{;ImB}g{WUZ>%gM*`4M_*he>3!2zyQ+OX6ninxyym6jn^zrRb(3MuH&4'
    'Sae8Tp!RSVc@!tWHY6BZ#ew(%jU>{<K_zitE%Wup%DqE=kPkfxhFiEIj%Nzy%#QsY?jh)J=qESypYTbnA!Q*{vjUw<LQ*5Z{jpp7y)*0!Iy#Ag'
    'HZ8peMmuQZO%ZmD#n`H+uo;VV1>d0Q8Mm!A%Q6A(EfQP<*oGN!d!g0<W+D|)jz*;dK@&h+12Rx-R-U&Mskio>XU>3RaE&r3$u@RXH+SOuCc<S_'
    '(4P1D-<l20Mc|Uth(=#U1?Vp@F=3gAA}oN0z<pU-gyIwI1At%W8?lE(0_CO%<pxUNRWi%7!Rv6b%%)`A;>`?M&_KGQ<smz1TITBF;_VfxUb2tt'
    'EFt1*YGwmc@VpiU&<v%iYi6~snQqc}DO&n-gZyaGS8<IaN#`OLDX65P#MRe8qS9JPy50btH&IUVzippoJ-1-bJhag(`|<lG0TQC2hjqX%<6KKn'
    'A0<$BnMJWAD?ApT5_|~O<Tk88p+DLI*oYPA`DK_rq%5h}3VZMX^A0m`n?MItm?VExim#?a1yVmke`pR*0V)=2k)pA2s|^6$$%4Eb)i^_5gU<vs'
    '3W9+T0{lC>E5ci`i~$R}SR`SUt)cOVo-o=9+vQVqxC&6FNa#poGQ|6cGzfvONaL5$Y7GX1y*M6u6)k}PE7G>OyLy|X05Cc5{HA2wEdVHZ4Hoxx'
    '+uz%)QF6TP9dkZ0{vC3?+S{E0Kh{+51h##gtfGZdCbc|EpUzwOgmA$K8we&QYSyk{HpPYDH|aM_YCmpt`R0e8KZX~VZ$JF}{_=XPT#Je3%zBJ3'
    'sQnM=y3{@k`zmr}v}Ez9FIxJ>_}@cP-m<4#Y^LapdH_3wf1u>pXD(om%a2h~h{F3dT5iNeo@cqQ;i#v!Jv~*k0~FF$?w&N&X+&CSJO*9VLeug2'
    'U@~c)|L0EBu(mMz?sF*Xkx^Clw9{yI+RHtr@@OmBZXjE1)jlhwlMA<E=*+V9V=oqBjb9}5`38`=LH)@Xh=NQ#2NhX*sI6T8kK#BO{)qiL3EBcw'
    'Rt|AE0yi!vtw;V_I0a=2Tck&#2SAOG6STY$aKi6!q$q$>6Q4K?lRE%xca@;;d5K}tI+_U&LfVDM*EtyB;WAoH<H(r>&VAsVp58s2J7{H=$+Rb3'
    'J2YRtc(Ym;9)J{CUWP!<Vu%{g?}5?><Ky{$?^b+@lLb=T_oay=<x7(pk^NK-?3kQ8@TVs`qUQ!?oudo0tJdhrr38#5aH4pXlsu@0oMOJQM2y@4'
    'qrK?$4G4uF+XnjvxCc!vVttGz+@Zl=s==S6Iy%SsLM-j8I!w7k>()tn_I1{aLd4AfJw!Yrf7C>&VcXDuK;LJktalU_(%^+zx_qSMjY{i)A}AnS'
    '-XN$hbzcvwA)rKvYq45`>y<yg1&cT<v;5KXov3hP^;088o@FSpAHbFMs&-AKp@5+f?dnGq#DhHBalEC(rdj+5J#T2u<j+u4;+;nMqVUhb0)nHF'
    'LewoCIY)+QH#h0ciIdD}j2`SyUKY44kK=IHcLZ2o$Nevdydvi|DPYbR_Y!5HyK7FwqcuJ!e17jC$BY_ozK1jPEzJwd9U~wzm!*qLJh(xuDJ}g|'
    'BUBy0gBCN|D|W|?NoX1l2luu9y!)iW5EI79Wll)_Kto!G4;>S3j^8=?I&Xs?+NQ6Ql(S8Zgiu#d1$G!BhoVGFjha@aC86;Wnsz#m)rO-Ecy>F>'
    'u&yyH&|``5nDGFu<yV4XgsIvFwc@eL9o585G=x#92a7p01#^KBs;AKQH1?)MlEu=8@D!p{Lyi>%CV*(G_8Ym*Z_l1N{|i*JHh8zQN5n7%V?D(Y'
    'BuWsOFP6#rEX4c}Mq1K=>#ZMwM^H0*e!D0g(_iCpcfSg_IjQuSh;L3%f~|40q6f=ciQ~r)oU|BvwC*0NTkIXCMc`DAR`QW=jrC-=8~=L}SM9Sl'
    'wLhGT=_Xl%5m+!#IM1_H<)am#gu;<-QJ&V*uo~KG(NcS$N1_r~uM)D#vuA`Z>k06)F>G}tW=FO<qYv+$RZ?Ky$8ES0b(Ru?;DRla_xlDq1_}d&'
    'N6~{$=6RAAW%Eoxf_q!gFN8tcO8U)y2DulK)NfJV`n|!c!12ft?CXCF?Mu1WrwytR@-mac&7lIVpm&c<RA3y@NBQO^9>{EK$4Q$b!DYZ(P>|DH'
    'HzxaW*;T;Mitv5m9b8?YP?jPBU7w@F=Kc!$X&)P}!IJ>7!BeYZEj^qh#iR*76EOBYOuaVU%F{t7hY+w9)cN}>p)?s^S=bMw>mS%ElERYoEOUX4'
    '5$xPLf^j^CuktE56y?yAY$<`0Lr=j|jlmD|%s>;-svU0F0?UTu*ytU2YP621Cve{U^6}%v<#qV|=*J%~KK>T|V{~;9j;^l5s~<*ysS48QD1h9C'
    'qbF%EkH}f|tI0!NWEjqqr3gb`r>C!{K?MB8=wJ@kfHDl;%?Y@93xOgooqV0yKENOKn84sSCIvrDCO0R6v%m9o1$=V?RPtPDbl&waDL|Q25424H'
    '-tlTU@F|b67XnCBMtE-O61wjkTXd}0qK7`o&B=FdlLIm8UGuzdPFzJ^JGstk!fx%w*=D)~KA>p=iJ2CiSo%Fl06fG3iN~WLuR8Lgh07M1S>y!*'
    '#!_I8GorKJbEnrI$dT;lXE1=SwRJ_4dTQGZ8EVg(sLbe)`#d#!(#_(wBm;}Cmzb3>crtJX6~j&=7&=K&4JZkef=j2|c$CdajPl$`k=4Dl_S4E9'
    '1wjzYBw!3-{~;Lm(ts4oyG=^D-M3;6CJ+ow+M}Rq2s)c_htXOd<a8>FN;_~5DYQI*{#>o1S{85dA2WN#eZ-_CpEV{epKFr_Q}b@KT+RhC+ec9('
    'F<>Hlb8?<SOK|cY)7c}+X0siVnRIntT`>XJl<%a7uA&q=<u@rBnJ#$4V1RcD|9{*7b#4W41o-$0j>v$?I;?7lAzFFFW0dEWgmG4{Aa>6-r5w*m'
    'Fjj}@GG>T<S>rB&Vi3ruMV9N)x^DErU4p;i90M~?9^GPO<trrQn8M0JI_3>ST1nuLqX8m?^$A0?9C7&w-&|mBh_6W5yf|}hsiJLET}fE6zx+-Z'
    'a2h&piMTEJ;ZtGR8Qj(t_O&yGJS%xPj|Z59)o(1X%&3|~-=>;gq{-{KH%UY5m9**TuG}<*q?!SZ4zrAT-FkOZY9v{BKRZV8K%Zi>D|d*As(wg{'
    'YNN@y_y@*}lUdn0A~w35R|T|VkI~=x)j&Q+op&$|R0%vV#B~_xPUs|L2r4u`I18`bz}^N__v3d>2gj6@?1fZW6NZ5^R#Tyw)YF*wx?JqT(A+fX'
    'H8plk;<0=%T%o7$bB7P6oW_V;%JJmrswhtKWB%g~6Cd?ao>wK(@l-V_G3`mk3^TEo2ZrcmfjZ%@vt%u_Q^fkDQzgZDq;WHObj(RzTemtho6K4k'
    'pibbgN7wJ)g#Wzw?P{m@P_72}^!Fk~h8b>MrftrpB{Gwuj`txd!Fs271EJ%P(j$1*@1IN2@*sA?zz*xJUcdCI)CQGWl9sPux_)PI0H7s}o6HlD'
    '4=Y?f#@5H`2f&F&id{_w!nIk-DFxWqFJ8S08si340t$lOhr;Y_H0?#AJjdY{IlEr-faYoF+$u?HKBwbH&qa@vN1<g^BUQQ|-Navy0s7z?1ZoGO'
    'gndy#wyex^MIYN+5a21zeIkHrTl}zBt!6lQbtZKcLTUT^>-&rUxl;KsRou9xW4%yNRaacsn3T7acZUbx?yLy0$%|jH^iZX4+#{<1|GFAQmRqeH'
    'i*+AA_O*hFQ3J*nD><O=o3(eKpQFu^j;yz*r%ETi9sNX$1YQC|5{C~WS=^QYJc}OHh8SoTIY4ri+0(F0v_)`0MA>NVDOQK%utQx79MJMKKHmVN'
    'f9v`>d8kOvK%ynZ<D|GVYdc+;>U<_sozK{@e-^Eu-4ERMvdjj%ou2YGVb<Jj2vWTt#<3_pc1=R;sWfEL=-4}mp$Y{V8cQPT&65WSz@_Fk(#poo'
    'O`ee2=N<&xpU^f^#Er=Mh)~u(%89|cdVM}eS%=ZRKIMLI&>)C}3X9(jz`g~!+o@v<EpU|Vfx9a85TB-)372;3tImRzk&x$cW>-L`TWLDMnbbp#'
    'tFsrg-fR+Z>qvJ3s<122=S`|Zd3hp@>AdG?7Ojf9B(t_~COQtxo#VTS?f}iO3Vy+bU-LQUP3k&Wa+cj&EUu+$(I|yjc_-Q1qHeW_z5FZdc<X{Z'
    'S-HG3S3bttcP#MIoF^uhg3zhoQSR5>tFG?8J8#+QyNRz;$f#%2PUGrgMeX#>oI-akqQ7e^Hd9(=NT-$<-xm4-CN-nD*hXL|CAVQ5CWc$C)rQnr'
    'G#No)c?cgH8KDNXLcl>f(~2v|6YoW9tPk|6B4`e6I2&i5Gw~28L1_Uz61x+xkJ+F`gkS^^`V!O?3{ntQG%SN&E*T%P%W*LFR{433{5>ri#AFM9'
    '!OVjeGhzdO!3NodM<+_rH%{AlX;#Xv=7ECVRizoAFXb14>gtEX3k7npXi7rK@S^~ZCD-Zn^ghbfQ3H18I|^huQ7oknnGoWN1yC&OUnJ;DthJ;('
    'K3!7a>LF&7W~C8t1!)CUX+#H)Y-$yQr=Tp+B^;Y_rXROz32y6{!2gUC8EZ9~o6HeaavRnb(3g=FMqD1UZn6}VIr}NY;D>4#hBSAM)x%_wX0Ut1'
    'F={vrjBu{1P5{lVzA5HDsf$E9#MqkRq{-y6Ny~Rf_gKcSlmt#gtaV;C1k#gHDr0l&e#mEpYFv9Y#)+7eL4EdzCSh7p498oUMJ{P*{0f}+m+vk<'
    'UR=Jp2$^w*wH>HgmukeS%U2f1sftVnVzq36co_LITDvTKuNA>NS-a|dg)WS>tS?%sDK!NJ^Ep)`m1+U~8qZ{Q0~J*ZeyO@_O@^ibV!WXd@C6R}'
    'fJ4I}1jvGCujoR`=Ugr+aOhHiYKsDATKV0PPa2PZOzQ+8Cs98{@cjHhP80-2L@PTng_YiRM`;>lzf;>Jwb-4?A_G2{ic=XE({L<7DOpypQ+8Z}'
    'W^Te0_>tQb4^y8tEfuo1h&@gcVAT#8s=(o6(mf`qO7vHC7qW=F3Xn<NAxZ?JLqW?!rE(_lVvecgU`qs3Lu7^@JU-lcVwB^^U|z4DYayHh|ICUy'
    'UbkIQhf4+Z4Ev7PHcV()D#P_yQ=MVBK+-qLu*SNNHz6Nf)?A_1$FIJA@5IVw(s-TaUpt)`T%c|Z))IK=5ZY8(xRJ~1V<tQqW`<Ml9J>H$L5W}r'
    'fFznINyUCjF&?ZL!~6-C$`LyExmGPeF{ipG5SpD@vKE~#tDYY?lnrU7eC&urqH<OEgV!*{BAPwg`A=KpzNyGg8g58`eZF)U=ZLDMQ>go=L;VNE'
    'P1!PbyBsN*mbJ7qx>AAE(IKAXwY2-~m3(;k;1!QWZ$&Omfp?3EX<!QUx$}c375#NVgR(5}j5?4K9@5-~hA!#sOn0I5x-N9r3Sf><fX_QChpb0C'
    '`;)B6+hE4<RbH^YpY8~o18Ob%mYfQ!A3d{e)ng)-adXF6x`vfGLLnXEA9ZtOs5}}V{1E=tDej7{f#IPI^&df%9vRx{h9Q3hZ19Hk7teni=ex29'
    '1<%Rz72b6h$+|2;aGo=+`Jt?Q$f1e6Q6|it)XR&$43lwu_}AToUjY36dtKW`;`k4|bX(oCt#98_HgW&8+wP0C{odc7a4q-Hwc8h6*KYsAZ`~eu'
    'kydh)+<7(unB7%#@D>3vx>nc)(xV-`j6VLPUkTWEqfA}hg1yU9%F9WBXtTgQ0duYOXuV>_SL)lYmr0DdSO5(vyVrd!Oiff~m0+sRh$rQBuQW>0'
    'iJ)+(r3kp~QrW_9g`)yZOkBf9b*W5NDA?f5`Y7l$gnxYg4*z7hFk~k`cyjgf{*9)4Xxdh5yBTPd@(vKYPC3fE$aL;^fEfe6XGx%vO9p)^)U|J$'
    '96|dh(ETvYKZDH2lh_cJ7Gf5%ObKYMUt-RqhY;_8hhW6-M6BPcW`L?b6<;|~fXXQGFx@?T6<-y8qogJ1J*!=n?NlNhu9q7Flq)3>T#k!{eQ9Z2'
    'c$P2F;5&!age%{yaZgHKh5iHX`F3y5_VjdNUHffrmMIJERhV34j?Byi>uS}uT=P1b&@TwOF_xa)he;)|>;Gj0BchlD+b64<jP};q+N<i<Kj%*3'
    'F#^}=TIzM?oaHOX8>}5gwnb)*`f+`_ZeMVLr2Qfv`(RC$V?UGoZk`4E`;m@*xrWH;{q=9*XQ1bh<%m)2gz1w;F-{8)4D^p*F5mw6S)*_cRvRMa'
    'lhsBU?AOiyw-m^tU8@J?vOaV8hiG}{;r#se!A77e7AcPtrH`Jcfs_n#Nl6t1D0@jv(;$0F1C=;X+T%E<@QrHyy-r<YTLjGBIIg((w3eF8!}091'
    '2>?frYAa)iGc$&Gp`YQ)Vj1|m!^FTDC_j*0Bgmqfb}5fk-p)x{lJoN;1F!jJxqM>ORmW<o=C7*`$?}J@`MeNBe4a@^px|Qg+Sar(_<#{;+%b1N'
    'm;Jwi9uv!n1?E2{c$A1OM@fRIZ1dK|wE*q$LyH(?XH4eNczU2m55_KApGo~H63VbvwHW*3Pk1PQpq~Ky5;8g&g3*&Bqg90URUJzehknj$pGG0y'
    '6=>E7P$OL+XnhTt^_DM=-SppQRVdBhjZnIZ3D01=`4L`-0%d1TUiwRQv<fTTHROaZ41z%fbD9cz)r>s+t%6xA9_2&`1zEL6Vq8D`Vr*?W6jCsM'
    'LgGZuoGV)Xyab7r7y1A9lKscNm&|6f{aHE*cQ6Md;o8Fy`MM!IXm7kFM+F+4q^m)WeM3(Z-iFY#C>*%7z0(JHe88!JcJ_ji?5SJ)RY4rL?Rt*k'
    '3F(NVBWCXMrw3paZ%`P<Z-27bgznjK#41m1rT|$NwD&@wTZyFrVMuBg>$mYTjt*?quN!@+c*=rrMUH@H{-HNCSxG3KKE`s!Aue6WYU+QoA{1WM'
    '_4@SFT7drrZ{RXE#28pk3tiR&mxZNKJ#?j*Zq1z=ZdGXak=LUMws}Dq!ocs}UygnZFMob}akZzNFAz5YuwQ4Ref_v);l?%Mj%AxyjrB!SRXZ8%'
    '{R_i1$bqG`foO|sYqplx26BA?&4=e=f*nA2U{(ZUPi?1ee&a0+BpBZw=^HQdA!<s<JFzaUV1|NYyb}r=Kv8e!RX2oeMx_>j3N@*2+YC?bc8pfl'
    '{F*m?aY^H5kBRe5^|~#pq5}<V^~FW10Ea?r!`7-=7k)Dfjw0XB08u{%&)dO{O6%|iL{^T0++nKl#zG&=cNW##30>{Q0zLVHB%H{Xh;a*>9Rli-'
    'S)0W=AX-FwUf<4b>NCEetAkjQ&j5Q}BB>q@y|mJ}ru(7J#yRp-*NNh24PL^3W4%=&VtJqI8NOj^cKbX|3c3%Du81?oqjB83Rdz4p>VmF1XKl}4'
    'E@am72oJKlt!uwj>j}Qa5r+eZQt4mk{NRs270`u$QN5!Jooh+?@lk1W?9^AUnu9*=iF$;B#^Ze7NjEaB)L*3nD&V#w06A`Zgg5Exs*e&`qi`-='
    'dxhJ<q#n3IWLC}7Sd67!HI8J@aR~e}ff$Yf%*S=aE0;x&Wc*h3{l;jmqT+>mvn)M+N}#2SRNx@>_MA~t@bZeB`g(P(&BnulI;`Q%f+yWOM}U;@'
    'ATXr=H;&tYYTS&@9*EQ1fH7qZUykX}$sY!f(3vQ{=OTMB?9t+6_kTLp6+{'
)

FULLFEATURE_TEST_ORIGINAL_SHA256 = 'a2fe1e0c521f1c36bc8cc002082e3ca58fc915f9cbfd487a5f537559276406c4'

FULLFEATURE_TEST_NODE_SHA256 = {'current_gallery_source_boundary': '063bd01f17a79bfd50b150ebaca3155f730aff0b6ae3927d73c9a7ac9c341106', 'current_gallery_test_boundary': '4ca5b233716c1472afac59a177acc329ed9a044cfe63cf23ae3586b5bec461ef', 'ContractTests.test_updated_A_current_bytes_binding': 'ffcb8af9f0dd4f6745f1dfe67dfe6ef1e3f6d1d96c1bfe107b840a91ca4ad5b6', 'ContractTests.test_terminal_rejects_partial_false_and_nonfinite_cpu_proof': '4fd680e514cc3da44035d2cdfd869c5c49ff03ae2c05be53b7206b15c1c7e8bf', 'SmoothAPTests.step': 'ecac4e0776ba954179b1065ea1ea0fff1b4e10d13afe8c6b870b255dd4bbe173', 'SmoothAPTests.test_cpu_witness_requires_positive_nonnearest_loss_and_total_difference': '96469f17afa607b803507352f33efcfed6f9d8c55c8eb74efe0b5b2025b47862', 'image_anchor_gradient_fixture': 'd195581ff0aef376448cccf79fe2046395c5bb142f21e2e9a895101dcd1e41c5', 'ImageAnchorTests.native_identity_boundary': '48acb11562baf6e4548d4088959347f5a7eeaa41454438e65aac3a7f6df9b58f', 'ImageAnchorTests.test_both_arm_receipts_zero_and_target_difference_are_authenticated': '23002a4d8a7fe3e0dbabadf6b8524e410dc2970be6c567cb557142b3cc55e569', 'ImageAnchorTests.test_prospective_schemas_and_both_arm_ranking': '5b98c6071070d56a97c8aa88e517fbedf93cd787f63ed571fb44b12bd5503abf', 'CurrentGalleryTests.test_candidate_rebuilds_gallery_from_current_same_A': '649d9a97099a1fc1557da58c54c9e7c79836f0541fbea50be2a45b3fdb62fb81', 'CurrentGalleryTests.test_active_objective_authentication_reads_source_only_and_rejects_substitution': '20ec77439b7b429449843c7e19482326b6dd31623af238e26c8ab9334d68cbbf', 'CurrentGalleryTests.test_precise_prospective_inverse_preserves_historical_source_and_test_guards': 'e0e09889cc3de3aff10f51021330cad7f3f394264baf632a615f1a4ef1b512c2', 'fullfeature_source_boundary': 'a5d37afdafa91893ec94277d2ff0a80939ef5a556dc1d779bfe1eb047c036a33', 'fullfeature_test_boundary': 'a5e5477926295384eaf3cc9b7b08e966594c731ae93430fd763085a7212dbc55', 'FullfeatureResidualTests': '92db1e0bf785124401e7d1e9019d86ba908fe5ead62161150c9cade5a16078ea'}

FULLFEATURE_TEST_BASE_AST_SHA256 = 'bc92da4982d960dea07a3da24e7662180da491dacc8568b0887bcd5fb93eb1bf'


def fullfeature_source_boundary(tree):
    """Invert only the declared residual delta, then retain every historical gate."""
    return _current_gallery_inverse(tree, FULLFEATURE_ORIGINAL_NODES, FULLFEATURE_ORIGINAL_SHA256,
                                    FULLFEATURE_NODE_SHA256, FULLFEATURE_BASE_AST_SHA256)


def fullfeature_test_boundary(tree):
    names = {'FULLFEATURE_ORIGINAL_NODES', 'FULLFEATURE_ORIGINAL_SHA256', 'FULLFEATURE_NODE_SHA256',
             'FULLFEATURE_BASE_AST_SHA256', 'FULLFEATURE_TEST_ORIGINAL_NODES', 'FULLFEATURE_TEST_ORIGINAL_SHA256',
             'FULLFEATURE_TEST_NODE_SHA256', 'FULLFEATURE_TEST_BASE_AST_SHA256'}
    counts = {name: 0 for name in names}
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            counts[node.targets[0].id] += 1
        else:
            kept.append(node)
    driver.require(set(counts.values()) == {1}, 'exact fullfeature pin constants required')
    tree.body = kept
    return _current_gallery_inverse(tree, FULLFEATURE_TEST_ORIGINAL_NODES, FULLFEATURE_TEST_ORIGINAL_SHA256,
                                    FULLFEATURE_TEST_NODE_SHA256, FULLFEATURE_TEST_BASE_AST_SHA256)


class FullfeatureResidualTests(unittest.TestCase):
    def test_exact_role_allocation_and_old_schema_rejection(self):
        self.assertTrue(hasattr(driver, 'parameter_roles'), 'missing fullfeature optimizer contract')
        self.assertEqual(driver.parameter_roles('control'), (['A'], [[128, 160]], 20480))
        self.assertEqual(driver.parameter_roles('candidate'), (['A', 'C'], [[128, 160], [128, 1152]], 167936))
        for invalid in ('concat', None, True):
            with self.assertRaises(ValueError): driver.parameter_roles(invalid)
        value, args = ContractTests().launch()
        driver.check_launch(value, args)
        for schema in ('siglip2-compact-current-gallery-smooth-ap-launch-v1', 'siglip2-compact-smooth-ap-launch-v1'):
            with self.assertRaises(ValueError): driver.check_launch({**value, 'schema': schema}, args)

    @contextmanager
    def residual_runtime(self):
        from contextlib import nullcontext
        class Tensor(GalleryTensor):
            trainable = False
            @property
            def requires_grad(self): return self.trainable
            @property
            def grad_fn(self): return None
            @property
            def is_leaf(self): return True
            def detach(self): return self
            def float(self): return self
            def __sub__(self, other):
                if len(self.shape) == 2 and len(other.shape) == 1:
                    return GalleryTensor([[x-y for x,y in zip(row,other.values,strict=True)] for row in self.values])
                return super().__sub__(other)
        def check(value, shape, device, frozen=False):
            driver.require(value.shape == shape and value.dtype == 'float32' and value.device is device and
                           (not frozen or not value.requires_grad), 'tensor shape/dtype/role differs')
        torch = SimpleNamespace(isfinite=lambda t: GalleryTensor(all(math.isfinite(float(v)) for v in t.flat())),
            count_nonzero=lambda t: GalleryTensor(sum(float(v) != 0 for v in t.flat())),
            autocast=lambda *a, **k: nullcontext())
        calls = []
        def linear(x, C):
            calls.append('C')
            return GalleryTensor([[sum(a*b for a,b in zip(row,w,strict=True)) for w in C.values] for row in x.values])
        def original(features, head, A, means, arm, primitive):
            calls.append('concat')
            self.assertEqual(arm, 'concat')
            return GalleryTensor([[.25 + .01*j for j in range(128)] for row in features.values])
        functional = SimpleNamespace(linear=linear)
        runtime = SimpleNamespace(Tensor=Tensor, torch=torch, primitive=SimpleNamespace(_check_tensor=check),
                                  readout=SimpleNamespace(raw_features=original), calls=calls)
        with patch.dict(sys.modules, {'torch': torch, 'torch.nn': SimpleNamespace(functional=functional)}):
            yield runtime

    def test_exact_formula_zero_parity_control_bypass_and_nonzero_mutants(self):
        with self.residual_runtime() as f:
            x=f.Tensor([[.02*j for j in range(1152)], [.01*j-.4 for j in range(1152)]])
            mu=f.Tensor([.005*j for j in range(1152)])
            C=f.Tensor([[0.]*1152 for _ in range(128)])
            def forward(arm='candidate', weight=C, mean=mu):
                return driver.fullfeature_raw_features(x,None,None,{},weight,mean,arm,f.primitive,f.readout)
            control=forward('control');self.assertEqual(f.calls,['concat'])
            f.calls.clear();candidate=forward();self.assertEqual(f.calls,['concat','C'])
            self.assertEqual(control.values,candidate.values)
            for j,row in enumerate(C.values): row[(j*7)%1152]=.2+.01*j
            f.calls.clear();actual=forward();self.assertEqual(f.calls,['concat','C'])
            expected=[[.25+.01*j + (.2+.01*j)*(row[(j*7)%1152]-mu.values[(j*7)%1152])
                       for j in range(128)] for row in x.values]
            self.assertEqual(actual.values,expected)
            self.assertNotEqual(actual.values,control.values,'omitted C must fail actual oracle')
            wrong=f.Tensor([v+.125 for v in mu.values])
            self.assertNotEqual(actual.values,forward(mean=wrong).values,'wrong mu must fail actual oracle')
            for arm,weight,mean in [('control',C,mu),('candidate',f.Tensor([[0.]*1152]),mu),
                                    ('candidate',C,f.Tensor([0.])),('candidate',C,f.Tensor([float('nan')]*1152))]:
                with self.subTest(arm=arm,shape=weight.shape),self.assertRaises(ValueError):forward(arm,weight,mean)
            zero=f.Tensor([[0.]*1152 for _ in range(128)]);zero.trainable=True
            with self.assertRaises(ValueError):forward('control',zero)

    def test_actual_public_inference_uses_loaded_nonzero_C_and_current_mean(self):
        from contextlib import nullcontext
        with self.residual_runtime() as f:
            class NativeTensor(f.Tensor):
                def cpu(self):return self
                def clone(self):return NativeTensor(copy.deepcopy(self.values))
            class Pixels:
                shape=(2,3,256,256);dtype='float32'
                def to(self,device):return self
                def flat(self):return [0.]
            pooled=NativeTensor([[1.]+[.01]*1151,[.7,.3]+[.02]*1150])
            class Model:
                training=False;_forward_hooks={};_forward_pre_hooks={};_backward_hooks={}
                def modules(self):return [self]
                def __call__(self,**kwargs):return SimpleNamespace(pooler_output=pooled)
            def normalize(x,dim):
                return NativeTensor([[v/math.sqrt(sum(t*t for t in row)) for v in row] for row in x.values])
            def packed(x):
                codes=NativeTensor([[round(v*127) for v in row] for row in x.values])
                inverse=NativeTensor([1./math.sqrt(sum(v*v for v in row)) for row in codes.values])
                return SimpleNamespace(codes=codes,inverse_norms=inverse,to_bytes=lambda:json.dumps(codes.values).encode())
            def fingerprint(value):
                def typed(v):
                    if isinstance(v,GalleryTensor):return ('tensor',v.shape,v.values)
                    if isinstance(v,dict):return {k:typed(x) for k,x in v.items()}
                    return v
                return hashlib.sha256(json.dumps(typed(value),sort_keys=True).encode()).hexdigest()
            f.torch.float32='float32';f.torch.float16='float16';f.torch.no_grad=lambda:nullcontext()
            f.torch.equal=lambda a,b:a.values==b.values
            f.torch.random=SimpleNamespace(get_rng_state=lambda:NativeTensor([1,2,3]))
            endpoint={'model':Model(),'processor_object':lambda **kw:{'pixel_values':Pixels()},
                'head_object':SimpleNamespace(state_dict=lambda:{}),'A':NativeTensor([[0.]*160 for _ in range(128)]),
                'C':NativeTensor([[.001*(j+1) if k==j else 0. for k in range(1152)] for j in range(128)]),
                'mu_train':NativeTensor([.0001]*1152),'mu_train_provenance':{'fixed':'TRAIN'},
                'arm':'candidate','means':{},'device':'cpu','flags':{'fixed':True},'modules':{
                    'qualify_siglip2_substrate_cpu.py':SimpleNamespace(numerical_flags=lambda:{'fixed':True}),
                    'prototype_residual_readout.py':f.readout,'quadratic_readout.py':f.primitive,
                    'joint_relational_compaction.py':SimpleNamespace(pack_int8_unit_embeddings=packed),
                    'train_siglip2_substrate_adaptation.py':SimpleNamespace(fingerprint=fingerprint)}}
            endpoint['A'].trainable=True;endpoint['C'].trainable=True
            endpoint['readout_sha256']=fingerprint(driver.inference_readout_tree(endpoint))
            functional=sys.modules['torch.nn'].functional;functional.normalize=normalize
            # Raw outputs are GalleryTensor; cpu is the only missing fixture operation.
            with patch.object(GalleryTensor,'cpu',lambda self:NativeTensor(self.values),create=True):
                output=driver.inference_outputs(endpoint,[object(),object()])
                features=normalize(pooled,1)
                base=[[.25+.01*j for j in range(128)] for _ in range(2)]
                explicit=[[base[i][j]+(.001*(j+1))*(features.values[i][j]-.0001)
                           for j in range(128)] for i in range(2)]
                self.assertEqual(output['raw'].values,explicit)
                self.assertNotEqual(output['raw'].values,base)
                self.assertEqual(set(output),{'raw','unit','codes','inverse_norms','wire'})
                for name in ('C','mu_train'):
                    value=endpoint[name];container=value.values[0] if len(value.shape)==2 else value.values
                    saved=container[0];container[0]+=.25
                    with self.assertRaisesRegex(ValueError,'current portable'):driver.inference_outputs(endpoint,[object(),object()])
                    container[0]=saved
                endpoint['C'].trainable=False
                with self.assertRaisesRegex(ValueError,'current portable'):driver.inference_outputs(endpoint,[object(),object()])

    def test_exact_inverse_keeps_original_hashes_and_rejects_every_declared_delta_mutation(self):
        tree=ast.parse(PATH.read_text())
        restored=fullfeature_source_boundary(copy.deepcopy(tree))
        self.assertEqual(hashlib.sha256(ast.dump(restored).encode()).hexdigest(),FULLFEATURE_BASE_AST_SHA256)
        completion_source_boundary(copy.deepcopy(tree))
        fullfeature_test_boundary(ast.parse(Path(__file__).read_text()))
        for name in ('fullfeature_raw_features','own_residual','check_optimizer','check_payload','inference_outputs',
                     'cpu_gradients','update','prepare_native','qualify_bundle'):
            mutant=copy.deepcopy(tree);node=next(n for n in mutant.body if isinstance(n,ast.FunctionDef) and n.name==name)
            node.body=[ast.Pass()]
            with self.subTest(name=name),self.assertRaises(ValueError):fullfeature_source_boundary(mutant)
        source=PATH.read_text()
        for before,after in [("raw = raw + F.linear(features.detach().float() - mu_train, C)","raw = raw"),
                             ("torch.optim.AdamW(members, **ADAM)","torch.optim.AdamW([A], **ADAM)"),
                             ("clip_grad_norm_(members, 1.","clip_grad_norm_([A], 1."),
                             ("views['canonical'].mean(dim=0)","views['augmented'].mean(dim=0)")]:
            self.assertIn(before,source)
            with self.assertRaises(ValueError):fullfeature_source_boundary(ast.parse(source.replace(before,after,1)))

    def test_mean_provenance_is_exact_train_domain_and_excludes_held_fitting(self):
        value={'domain':'actual CPU-renormalized FP32 features','rows':6355,'view':'canonical',
            'reduction':'torch FP32 mean(dim=0) canonical ordinal order',
            **{k:'a'*64 for k in ('canonical_features_sha256','original_rows_sha256','target_sha256','partition_sha256')},
            'accepted_checkpoint':copy.deepcopy(driver.ACCEPTED['checkpoint']),
            'canonical_cache':{'normalized':True,'raw_pooled_cache':False,'shape':[6355,1152],'dtype':'float32'}}
        driver.check_mu_train_provenance(value)
        for key,bad in [('rows',True),('rows',6354),('view','selection'),('domain','raw pooled'),
                        ('reduction','FP64'),('accepted_checkpoint',{}),('target_sha256','stale'),
                        ('canonical_cache',{'normalized':True})]:
            with self.subTest(key=key),self.assertRaises(ValueError):driver.check_mu_train_provenance({**value,key:bad})
        with self.assertRaises(ValueError):driver.check_mu_train_provenance({**value,'held':True})

    def test_ordered_optimizer_moments_roles_steps_and_control_exclusion(self):
        class Tensor:
            requires_grad=False;grad_fn=None;device=SimpleNamespace(type='cpu');dtype='float32'
            def __init__(self,shape,value=0.):self.shape,self.value=shape,value
            def __float__(self):return float(self.value)
        fake=SimpleNamespace(float32='float32',isfinite=lambda t:GalleryTensor(math.isfinite(t.value)))
        with patch.dict(sys.modules,{'torch':fake}):
            for arm in driver.ARMS:
                names,shapes,_=driver.parameter_roles(arm)
                ident={'arm':arm,'parameter_names':names,'parameter_shapes':shapes,'optimizer_groups':[driver.ADAM],
                       'device':'cpu','initial_scaler':{}}
                saved={'scaler':{},'optimizer':{'param_groups':[{**driver.ADAM,'params':list(range(len(names)))}],
                    'state':{i:{'step':Tensor((),17.),'exp_avg':Tensor(tuple(shape)),
                                'exp_avg_sq':Tensor(tuple(shape))} for i,shape in enumerate(shapes)}}}
                driver.check_optimizer(saved,ident,17)
                mutants=[]
                wrong=copy.deepcopy(saved);wrong['optimizer']['param_groups'][0]['params']=[1,0] if arm=='candidate' else [0,1];mutants.append(wrong)
                wrong=copy.deepcopy(saved);wrong['optimizer']['state'][0]['step'].value=16.;mutants.append(wrong)
                wrong=copy.deepcopy(saved);wrong['optimizer']['state'][0]['exp_avg'].value=float('nan');mutants.append(wrong)
                wrong=copy.deepcopy(saved);wrong['optimizer']['state'][1]={'step':Tensor((),17.)};mutants.append(wrong)
                for wrong in mutants:
                    if arm=='control' and wrong['optimizer']['param_groups'][0]['params']==[0] and wrong==saved:continue
                    with self.assertRaises((ValueError,KeyError)):driver.check_optimizer(wrong,ident,17)
                initial=copy.deepcopy(saved);initial['optimizer']['state']={};driver.check_optimizer(initial,ident,0)
                with self.assertRaises(ValueError):driver.check_optimizer(saved,ident,0)

    def test_current_C_mean_roles_and_source_substitutions_do_not_use_version_cache(self):
        with self.residual_runtime() as f:
            def digest(context,value,**kw):
                if isinstance(value,GalleryTensor):value={'shape':value.shape,'values':value.values,'dtype':value.dtype}
                return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            mu=f.Tensor([0.]*1152);provenance={'fixed':'TRAIN canonical'}
            for arm in driver.ARMS:
                C=f.Tensor([[0.]*1152 for _ in range(128)]);C.trainable=arm=='candidate'
                context={'legacy':{'quadratic':f.primitive},'nearest':SimpleNamespace(fingerprint=digest),
                    'initial_C_sha256':digest(None,C),'mu_train_sha256':digest(None,mu),
                    'mu_train_provenance_sha256':digest(None,provenance)}
                state={'A':f.Tensor([0.]),'C':C,'mu_train':mu,'mu_train_provenance':provenance,'arm':arm,'counter':0}
                driver.own_residual(context,state,admit=True)
                if arm=='candidate':C.values[0][0]=.1
                state['counter']=1;driver.own_residual(context,state,advanced=True)
                driver.own_residual(context,state)
                for value in (C,mu):
                    container=value.values[0] if len(value.shape)==2 else value.values
                    prior=container[0];container[0]+=.25
                    with self.assertRaises(ValueError):driver.own_residual(context,state)
                    container[0]=prior;driver.own_residual(context,state)
                C.trainable=not C.trainable
                with self.assertRaises(ValueError):driver.own_residual(context,state)
                C.trainable=arm=='candidate'
                prior=C.values[0][0];C.values[0][0]=0.
                if arm=='candidate':
                    with self.assertRaises(ValueError):driver.own_residual(context,state)
                C.values[0][0]=prior
                state['mu_train_provenance']={'fixed':'selection'}
                with self.assertRaises(ValueError):driver.own_residual(context,state)

    def test_A_C_decomposition_receipts_reject_each_detach_and_stale_role(self):
        bank=SmoothAPTests().bank();batch=list(range(12,76));membership=driver.ranking_membership(bank,batch)
        g=image_anchor_gradient_fixture({'seed':179061,'batch':batch,'membership_sha256':driver.json_sha256(membership),
            'mse':1.,'rank':.1,'K':64,'active':128,'control_gradient_norm':1.,'candidate_gradient_norm':1.,
            'ranking_gradient_norm':.2,'candidate_minus_control_gradient_norm':0.,'gradient_alignment':.2,
            'multi_positive_anchors':64,'nonnearest_positive_terms':2*sum(len(p)-1 for p in membership['positive']),
            'nonnearest_loss':.08,'nonnearest_gradient_norm':.1,'native_mask_self_ties_singletons_exact':True,
            'micro16_global_reduction_exact':True})
        driver.check_cpu_gradient(g,bank)
        for key in g['roles']:
            for field,bad in [('query_gradient_norm',0.),('gallery_gradient_norm',0.),('tied_gradient_norm',float('nan')),
                              ('detached_gallery_mutant_rejected',False),('full64_micro16_exact',False),
                              ('query_plus_gallery_exact',False),('gallery_gradient_sha256','wrong')]:
                mutant=copy.deepcopy(g);mutant['roles'][key][field]=bad
                with self.subTest(role=key,field=field),self.assertRaises(ValueError):driver.check_cpu_gradient(mutant,bank)
            mutant=copy.deepcopy(g);mutant['roles'].pop(key)
            with self.assertRaises(ValueError):driver.check_cpu_gradient(mutant,bank)
        for field,bad in [('initial_losses_and_A_gradients_matched',False),('candidate_minus_control_gradient_norm',.1),
                          ('candidate_C_gradient_norm',0.),('nonzero_C_oracle_exact',False),
                          ('omitted_C_mutant_rejected',False),('wrong_mu_mutant_rejected',False)]:
            with self.assertRaises(ValueError):driver.check_cpu_gradient({**g,field:bad},bank)

    def test_public_residual_helper_is_available_without_native_imports(self):
        self.assertTrue(hasattr(driver, 'fullfeature_raw_features'), 'missing public C/mu readout')
        self.assertFalse(any(name.split('.')[0] in driver.NATIVE for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
