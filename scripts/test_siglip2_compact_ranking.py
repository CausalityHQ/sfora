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


def smooth_ap_source_boundary(tree):
    """Exact named-node inverse; reject any unreviewed objective/guard edit."""
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
                'scale': 128, 'gradient_norm': 1., 'ranking_gradient_norm': .2,
                'A_before_sha256': 'a'*64, 'A_after_sha256': 'b'*64,
                'core_seconds': .1, 'seconds': .2, 'mse': .2, 'rank': .1, 'loss': .3,
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
        driver.check_cpu_gradient(witness, bank)
        for key, bad in (('nonnearest_loss', 0.), ('nonnearest_gradient_norm', 0.),
                         ('candidate_minus_control_gradient_norm', 0.), ('candidate_gradient_norm', float('nan')),
                         ('gradient_alignment', 1.01), ('multi_positive_anchors', 0),
                         ('nonnearest_positive_terms', 0), ('native_mask_self_ties_singletons_exact', False),
                         ('micro16_global_reduction_exact', False), ('candidate_minus_control_equals_rank', False)):
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



if __name__ == "__main__":
    unittest.main()
