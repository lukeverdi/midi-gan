σή
¨ύ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d388Έ
u
dense/kernelVarHandleOp*
shape:	d*
dtype0*
_output_shapes
: *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	d
m

dense/biasVarHandleOp*
shape:*
_output_shapes
: *
shared_name
dense/bias*
dtype0
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0

conv2d/kernelVarHandleOp*
dtype0*
shape:*
shared_nameconv2d/kernel*
_output_shapes
: 
y
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*(
_output_shapes
:*
dtype0
o
conv2d/biasVarHandleOp*
shape:*
dtype0*
shared_nameconv2d/bias*
_output_shapes
: 
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes	
:

batch_normalization/gammaVarHandleOp*
dtype0*
shape:*
_output_shapes
: **
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes	
:

batch_normalization/betaVarHandleOp*
dtype0*
shape:*)
shared_namebatch_normalization/beta*
_output_shapes
: 

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes	
:

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*0
shared_name!batch_normalization/moving_mean*
shape:

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes	
:

#batch_normalization/moving_varianceVarHandleOp*4
shared_name%#batch_normalization/moving_variance*
dtype0*
_output_shapes
: *
shape:

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
shape:* 
shared_nameconv2d_1/kernel*
dtype0
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:*
dtype0
s
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
_output_shapes
: *
shape:*
dtype0
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*
dtype0*
shape:

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0

batch_normalization_1/betaVarHandleOp*
dtype0*
shape:*+
shared_namebatch_normalization_1/beta*
_output_shapes
: 

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes	
:

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean*
dtype0*
shape:

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_1/moving_variance*
shape:*
_output_shapes
: *
dtype0

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:

conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*
shape:*
dtype0*
_output_shapes
: 
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:*
dtype0
s
conv2d_2/biasVarHandleOp*
dtype0*
shape:*
shared_nameconv2d_2/bias*
_output_shapes
: 
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes	
:

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*,
shared_namebatch_normalization_2/gamma*
shape:

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes	
:

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
shape:*
dtype0*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes	
:

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
shape:*2
shared_name#!batch_normalization_2/moving_mean*
dtype0

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes	
:
£
%batch_normalization_2/moving_varianceVarHandleOp*
dtype0*6
shared_name'%batch_normalization_2/moving_variance*
shape:*
_output_shapes
: 

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes	
:

conv2d_3/kernelVarHandleOp* 
shared_nameconv2d_3/kernel*
_output_shapes
: *
dtype0*
shape:@
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*'
_output_shapes
:@
r
conv2d_3/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0

batch_normalization_3/gammaVarHandleOp*
shape:@*
_output_shapes
: *
dtype0*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:@

batch_normalization_3/betaVarHandleOp*+
shared_namebatch_normalization_3/beta*
dtype0*
shape:@*
_output_shapes
: 

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:@

!batch_normalization_3/moving_meanVarHandleOp*2
shared_name#!batch_normalization_3/moving_mean*
dtype0*
shape:@*
_output_shapes
: 

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:@
’
%batch_normalization_3/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0*
shape:@

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:@

conv2d_4/kernelVarHandleOp*
dtype0* 
shared_nameconv2d_4/kernel*
_output_shapes
: *
shape:@
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:

NoOpNoOp
S
ConstConst"/device:CPU:0*
_output_shapes
: *ΌR
value²RB―R B¨R

layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
R
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api

3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
R
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api

Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
R
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
R
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api

aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
fregularization_losses
g	variables
htrainable_variables
i	keras_api
R
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
h

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api

xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}regularization_losses
~	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
Ψ
0
 1
-2
.3
44
55
66
77
D8
E9
K10
L11
M12
N13
[14
\15
b16
c17
d18
e19
r20
s21
y22
z23
{24
|25
26
27
 

0
 1
-2
.3
44
55
D6
E7
K8
L9
[10
\11
b12
c13
r14
s15
y16
z17
18
19

	variables
regularization_losses
layers
non_trainable_variables
metrics
 layer_regularization_losses
trainable_variables
 
 
 
 

regularization_losses
layers
	variables
non_trainable_variables
metrics
 layer_regularization_losses
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1

!regularization_losses
layers
"	variables
non_trainable_variables
metrics
 layer_regularization_losses
#trainable_variables
 
 
 

%regularization_losses
layers
&	variables
non_trainable_variables
metrics
 layer_regularization_losses
'trainable_variables
 
 
 

)regularization_losses
layers
*	variables
 non_trainable_variables
‘metrics
 ’layer_regularization_losses
+trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1

/regularization_losses
£layers
0	variables
€non_trainable_variables
₯metrics
 ¦layer_regularization_losses
1trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
62
73

40
51

8regularization_losses
§layers
9	variables
¨non_trainable_variables
©metrics
 ͺlayer_regularization_losses
:trainable_variables
 
 
 

<regularization_losses
«layers
=	variables
¬non_trainable_variables
­metrics
 ?layer_regularization_losses
>trainable_variables
 
 
 

@regularization_losses
―layers
A	variables
°non_trainable_variables
±metrics
 ²layer_regularization_losses
Btrainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1

Fregularization_losses
³layers
G	variables
΄non_trainable_variables
΅metrics
 Άlayer_regularization_losses
Htrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1
M2
N3

K0
L1

Oregularization_losses
·layers
P	variables
Έnon_trainable_variables
Ήmetrics
 Ίlayer_regularization_losses
Qtrainable_variables
 
 
 

Sregularization_losses
»layers
T	variables
Όnon_trainable_variables
½metrics
 Ύlayer_regularization_losses
Utrainable_variables
 
 
 

Wregularization_losses
Ώlayers
X	variables
ΐnon_trainable_variables
Αmetrics
 Βlayer_regularization_losses
Ytrainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1

]regularization_losses
Γlayers
^	variables
Δnon_trainable_variables
Εmetrics
 Ζlayer_regularization_losses
_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1
d2
e3

b0
c1

fregularization_losses
Ηlayers
g	variables
Θnon_trainable_variables
Ιmetrics
 Κlayer_regularization_losses
htrainable_variables
 
 
 

jregularization_losses
Λlayers
k	variables
Μnon_trainable_variables
Νmetrics
 Ξlayer_regularization_losses
ltrainable_variables
 
 
 

nregularization_losses
Οlayers
o	variables
Πnon_trainable_variables
Ρmetrics
 ?layer_regularization_losses
ptrainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1

r0
s1

tregularization_losses
Σlayers
u	variables
Τnon_trainable_variables
Υmetrics
 Φlayer_regularization_losses
vtrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

y0
z1
{2
|3

y0
z1

}regularization_losses
Χlayers
~	variables
Ψnon_trainable_variables
Ωmetrics
 Ϊlayer_regularization_losses
trainable_variables
 
 
 
‘
regularization_losses
Ϋlayers
	variables
άnon_trainable_variables
έmetrics
 ήlayer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
‘
regularization_losses
ίlayers
	variables
ΰnon_trainable_variables
αmetrics
 βlayer_regularization_losses
trainable_variables
 
 
 
‘
regularization_losses
γlayers
	variables
δnon_trainable_variables
εmetrics
 ζlayer_regularization_losses
trainable_variables

0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
8
60
71
M2
N3
d4
e5
{6
|7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

d0
e1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

{0
|1
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0
~
serving_default_dense_inputPlaceholder*'
_output_shapes
:?????????d*
shape:?????????d*
dtype0
ύ
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/bias*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:????????? *(
Tin!
2*-
f(R&
$__inference_signature_wrapper_322514*-
_gradient_op_typePartitionedCall-323465
O
saver_filenamePlaceholder*
dtype0*
shape: *
_output_shapes
: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2**
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_323514*-
_gradient_op_typePartitionedCall-323515*
_output_shapes
: 
Χ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/bias*
Tout
2*-
_gradient_op_typePartitionedCall-323612*+
f&R$
"__inference__traced_restore_323611*
_output_shapes
: *(
Tin!
2**
config_proto

GPU 

CPU2J 8ζ
¦
ͺ
)__inference_conv2d_4_layer_call_fn_321931

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+???????????????????????????*-
_gradient_op_typePartitionedCall-321926*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
§
ͺ
)__inference_conv2d_3_layer_call_fn_321765

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*A
_output_shapes/
-:+???????????????????????????@*-
_gradient_op_typePartitionedCall-321760
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 


Ϋ
B__inference_conv2d_layer_call_and_return_conditional_losses_321184

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp¬
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*
strides
*B
_output_shapes0
.:,???????????????????????????*
T0‘
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????€
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
a

F__inference_sequential_layer_call_and_return_conditional_losses_322303

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCallό
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_321948**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-321954*
Tout
2Κ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:?????????*
Tout
2*-
_gradient_op_typePartitionedCall-321985*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_321979β
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-321168*R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162Ί
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-321190**
config_proto

GPU 

CPU2J 8*
Tin
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_321184Ω
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*-
_gradient_op_typePartitionedCall-321298**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*
Tout
2*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321297ς
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-322030**
config_proto

GPU 

CPU2J 8*
Tout
2*B
_output_shapes0
.:,???????????????????????????*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024κ
up_sampling2d_1/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352*
Tin
2*-
_gradient_op_typePartitionedCall-321358*B
_output_shapes0
.:,???????????????????????????Δ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321380*B
_output_shapes0
.:,???????????????????????????η
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tout
2*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321487*
Tin	
2*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321488ψ
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tout
2*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-322075*
Tin
2μ
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-321548*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*B
_output_shapes0
.:,???????????????????????????Δ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-321570*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8η
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tin	
2*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321677*-
_gradient_op_typePartitionedCall-321678*
Tout
2ψ
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-322120*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114*
Tout
2*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8μ
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321738*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732*
Tin
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2Γ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tout
2*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754*A
_output_shapes/
-:+???????????????????????????@*
Tin
2*-
_gradient_op_typePartitionedCall-321760**
config_proto

GPU 

CPU2J 8ζ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tout
2*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321868*
Tin	
2*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321867χ
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tout
2*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159*-
_gradient_op_typePartitionedCall-322165*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*
Tin
2Α
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-321926**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????δ
activation/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-322186*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_322180*A
_output_shapes/
-:+???????????????????????????
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 

H
,__inference_leaky_re_lu_layer_call_fn_323095

inputs
identityΈ
PartitionedCallPartitionedCallinputs*
Tin
2*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024*
Tout
2*-
_gradient_op_typePartitionedCall-322030**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs
Ψδ
₯
F__inference_sequential_layer_call_and_return_conditional_losses_322732

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resourceD
@batch_normalization_assignmovingavg_read_readvariableop_resourceF
Bbatch_normalization_assignmovingavg_1_read_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceF
Bbatch_normalization_1_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceF
Bbatch_normalization_2_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceF
Bbatch_normalization_3_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity’7batch_normalization/AssignMovingAvg/AssignSubVariableOp’7batch_normalization/AssignMovingAvg/Read/ReadVariableOp’2batch_normalization/AssignMovingAvg/ReadVariableOp’9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp’9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp’4batch_normalization/AssignMovingAvg_1/ReadVariableOp’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp’9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp’4batch_normalization_1/AssignMovingAvg/ReadVariableOp’;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp’;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp’6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp’9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp’4batch_normalization_2/AssignMovingAvg/ReadVariableOp’;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp’;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp’6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp’$batch_normalization_2/ReadVariableOp’&batch_normalization_2/ReadVariableOp_1’9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp’9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp’4batch_normalization_3/AssignMovingAvg/ReadVariableOp’;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp’;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp’6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp’$batch_normalization_3/ReadVariableOp’&batch_normalization_3/ReadVariableOp_1’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’conv2d_3/BiasAdd/ReadVariableOp’conv2d_3/Conv2D/ReadVariableOp’conv2d_4/BiasAdd/ReadVariableOp’conv2d_4/Conv2D/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp―
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	dv
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:?????????*
T0]

dense/ReluReludense/BiasAdd:output:0*(
_output_shapes
:?????????*
T0U
reshape/ShapeShapedense/Relu:activations:0*
_output_shapes
:*
T0e
reshape/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0ω
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: Y
reshape/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :Ρ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
_output_shapes
:*
T0
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*0
_output_shapes
:?????????*
T0[
up_sampling2d/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:k
!up_sampling2d/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:m
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#up_sampling2d/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
T0*
_output_shapes
:*
Index0d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:Ι
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
half_pixel_centers(*0
_output_shapes
:?????????*
T0Ί
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:έ
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*
strides
*0
_output_shapes
:?????????―
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????b
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
b
 batch_normalization/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: Ή
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0½
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:\
batch_normalization/ConstConst*
dtype0*
valueB *
_output_shapes
: ^
batch_normalization/Const_1Const*
valueB *
dtype0*
_output_shapes
: λ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*
U0*
epsilon%o:*
T0*P
_output_shapes>
<:?????????:::::`
batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL?γ
7batch_normalization/AssignMovingAvg/Read/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
,batch_normalization/AssignMovingAvg/IdentityIdentity?batch_normalization/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0θ
)batch_normalization/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
valueB
 *  ?*
_output_shapes
: £
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: *J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ώ
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
T0ͺ
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
dtype0η
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0£
.batch_normalization/AssignMovingAvg_1/IdentityIdentityAbatch_normalization/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:μ
+batch_normalization/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
_output_shapes
: *L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0©
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
T0
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ι
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes	
:*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp²
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp~
leaky_re_lu/LeakyRelu	LeakyRelu(batch_normalization/FusedBatchNormV3:y:0*0
_output_shapes
:?????????h
up_sampling2d_1/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
_output_shapes
:*
T0m
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0o
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0o
%up_sampling2d_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
T0f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
_output_shapes
:*
T0Ψ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#leaky_re_lu/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(Ύ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:γ
conv2d_1/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:?????????*
strides
*
paddingSAME*
T0³
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????d
"batch_normalization_1/LogicalAnd/xConst*
value	B
 Z*
_output_shapes
: *
dtype0
d
"batch_normalization_1/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
:  
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: ½
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Α
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:^
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
batch_normalization_1/Const_1Const*
_output_shapes
: *
valueB *
dtype0χ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
epsilon%o:*
T0*
U0*P
_output_shapes>
<:?????????:::::b
batch_normalization_1/Const_2Const*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: η
9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:£
.batch_normalization_1/AssignMovingAvg/IdentityIdentityAbatch_normalization_1/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:μ
+batch_normalization_1/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ?*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp«
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: *L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource:^batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Η
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:²
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_1_assignmovingavg_read_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*L
_classB
@>loc:@batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 λ
;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0§
0batch_normalization_1/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0π
-batch_normalization_1/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
_output_shapes
: *
dtype0*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp±
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
T0€
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ρ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
T0Ί
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:£
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_1_assignmovingavg_1_read_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *N
_classD
B@loc:@batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????j
up_sampling2d_2/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
_output_shapes
:*
T0m
#up_sampling2d_2/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:o
%up_sampling2d_2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
T0*
_output_shapes
:*
Index0f
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
_output_shapes
:*
T0Ϊ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_1/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*0
_output_shapes
:?????????*
T0*
half_pixel_centers(Ύ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0γ
conv2d_2/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:?????????*
paddingSAME*
T0*
strides
³
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????d
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
d
"batch_normalization_2/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: ½
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Α
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:^
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
batch_normalization_2/Const_1Const*
dtype0*
valueB *
_output_shapes
: χ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
U0*
T0*P
_output_shapes>
<:?????????:::::*
epsilon%o:b
batch_normalization_2/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *ΝΜL?η
9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0£
.batch_normalization_2/AssignMovingAvg/IdentityIdentityAbatch_normalization_2/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:μ
+batch_normalization_2/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
valueB
 *  ?«
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
T0
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource:^batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Η
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:²
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes	
:
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_2_assignmovingavg_read_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 λ
;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0§
0batch_normalization_2/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:π
-batch_normalization_2/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp±
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: *N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp€
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ρ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:Ί
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:£
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_2_assignmovingavg_1_read_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 *
dtype0
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????j
up_sampling2d_3/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:m
#up_sampling2d_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:o
%up_sampling2d_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:o
%up_sampling2d_3/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
T0f
up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:Ϊ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_2/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:????????? *
half_pixel_centers(½
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*'
_output_shapes
:@*
dtype0β
conv2d_3/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? @*
paddingSAME*
strides
²
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? @*
T0d
"batch_normalization_3/LogicalAnd/xConst*
dtype0
*
value	B
 Z*
_output_shapes
: d
"batch_normalization_3/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
:  
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: Ό
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@ΐ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0^
batch_normalization_3/ConstConst*
valueB *
_output_shapes
: *
dtype0`
batch_normalization_3/Const_1Const*
_output_shapes
: *
valueB *
dtype0ς
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0$batch_normalization_3/Const:output:0&batch_normalization_3/Const_1:output:0*
epsilon%o:*
T0*K
_output_shapes9
7:????????? @:@:@:@:@:*
U0b
batch_normalization_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL?ζ
9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@’
.batch_normalization_3/AssignMovingAvg/IdentityIdentityAbatch_normalization_3/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@μ
+batch_normalization_3/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
valueB
 *  ?*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: «
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0&batch_normalization_3/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
: 
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource:^batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ζ
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:@±
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_3_assignmovingavg_read_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *L
_classB
@>loc:@batch_normalization_3/AssignMovingAvg/Read/ReadVariableOpκ
;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@¦
0batch_normalization_3/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0π
-batch_normalization_3/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *  ?*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0±
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0&batch_normalization_3/Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: £
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Π
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0Ή
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
T0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp£
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_3_assignmovingavg_1_read_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*N
_classD
B@loc:@batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:????????? @Ό
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@Κ
conv2d_4/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:????????? *
T0²
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
activation/TanhTanhconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:????????? «
IdentityIdentityactivation/Tanh:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*/
_output_shapes
:????????? *
T0"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_3/AssignMovingAvg/Read/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_2/AssignMovingAvg_1/Read/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_3/AssignMovingAvg_1/Read/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp2v
9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_2/AssignMovingAvg/Read/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/Read/ReadVariableOp7batch_normalization/AssignMovingAvg/Read/ReadVariableOp2L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_1/AssignMovingAvg/Read/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_1/AssignMovingAvg_1/Read/ReadVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 

τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
is_training( *
epsilon%o:J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *ΝΜL?α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
ψ/

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323145

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
epsilon%o:*b
_output_shapesP
N:,???????????????????????????:::::*
U0L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *ΝΜL?»
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
valueB
 *  ?Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
: ά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:*
T0Ϊ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ώ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
valueB
 *  ?Ω
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0*
_output_shapes
: β
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:β
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
Ρ
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_323190

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs


έ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp¬
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
‘
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????€
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
α
?
6__inference_batch_normalization_1_layer_call_fn_323185

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321522*B
_output_shapes0
.:,???????????????????????????*
Tout
2*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321521
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 

J
.__inference_leaky_re_lu_3_layer_call_fn_323395

inputs
identityΉ
PartitionedCallPartitionedCallinputs*
Tin
2*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322165*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159*
Tout
2z
IdentityIdentityPartitionedCall:output:0*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:& "
 
_user_specified_nameinputs
°Α
Ε
F__inference_sequential_layer_call_and_return_conditional_losses_322892

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource
identity’3batch_normalization/FusedBatchNormV3/ReadVariableOp’5batch_normalization/FusedBatchNormV3/ReadVariableOp_1’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’5batch_normalization_1/FusedBatchNormV3/ReadVariableOp’7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’5batch_normalization_2/FusedBatchNormV3/ReadVariableOp’7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_2/ReadVariableOp’&batch_normalization_2/ReadVariableOp_1’5batch_normalization_3/FusedBatchNormV3/ReadVariableOp’7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_3/ReadVariableOp’&batch_normalization_3/ReadVariableOp_1’conv2d/BiasAdd/ReadVariableOp’conv2d/Conv2D/ReadVariableOp’conv2d_1/BiasAdd/ReadVariableOp’conv2d_1/Conv2D/ReadVariableOp’conv2d_2/BiasAdd/ReadVariableOp’conv2d_2/Conv2D/ReadVariableOp’conv2d_3/BiasAdd/ReadVariableOp’conv2d_3/Conv2D/ReadVariableOp’conv2d_4/BiasAdd/ReadVariableOp’conv2d_4/Conv2D/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp―
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	d*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*(
_output_shapes
:?????????*
T0­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*(
_output_shapes
:?????????*
T0U
reshape/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:g
reshape/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0ω
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: Y
reshape/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0Y
reshape/Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: Z
reshape/Reshape/shape/3Const*
value
B :*
_output_shapes
: *
dtype0Ρ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapedense/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????[
up_sampling2d/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:k
!up_sampling2d/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:m
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:d
up_sampling2d/ConstConst*
_output_shapes
:*
valueB"      *
dtype0
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
_output_shapes
:*
T0Ι
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(Ί
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0έ
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:?????????*
T0*
strides
*
paddingSAME―
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:?????????*
T0b
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
b
 batch_normalization/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: Ή
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0½
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ϋ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ί
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:°
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
epsilon%o:*
is_training( *P
_output_shapes>
<:?????????:::::*
U0^
batch_normalization/ConstConst*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0~
leaky_re_lu/LeakyRelu	LeakyRelu(batch_normalization/FusedBatchNormV3:y:0*0
_output_shapes
:?????????h
up_sampling2d_1/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:m
#up_sampling2d_1/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:o
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0o
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
_output_shapes
:*
T0*
Index0f
up_sampling2d_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"      
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:Ψ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#leaky_re_lu/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(Ύ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0γ
conv2d_1/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingSAME*0
_output_shapes
:?????????*
T0*
strides
³
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:?????????*
T0d
"batch_normalization_1/LogicalAnd/xConst*
value	B
 Z *
_output_shapes
: *
dtype0
d
"batch_normalization_1/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
:  
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: ½
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Α
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ί
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:γ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ό
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( `
batch_normalization_1/ConstConst*
dtype0*
valueB
 *ΝΜL?*
_output_shapes
: 
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????j
up_sampling2d_2/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
_output_shapes
:*
T0m
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%up_sampling2d_2/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:o
%up_sampling2d_2/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:f
up_sampling2d_2/ConstConst*
valueB"      *
_output_shapes
:*
dtype0
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
_output_shapes
:*
T0Ϊ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_1/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(Ύ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:γ
conv2d_2/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingSAME*0
_output_shapes
:?????????³
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:?????????*
T0d
"batch_normalization_2/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z d
"batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
:  
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: ½
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Α
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ί
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:γ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ό
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%o:*P
_output_shapes>
<:?????????:::::*
U0*
is_training( *
T0`
batch_normalization_2/ConstConst*
valueB
 *ΝΜL?*
_output_shapes
: *
dtype0
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????j
up_sampling2d_3/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
_output_shapes
:*
T0m
#up_sampling2d_3/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0o
%up_sampling2d_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:o
%up_sampling2d_3/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
_output_shapes
:*
T0*
Index0f
up_sampling2d_3/ConstConst*
_output_shapes
:*
valueB"      *
dtype0
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
_output_shapes
:*
T0Ϊ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_2/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*
half_pixel_centers(*0
_output_shapes
:????????? ½
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*'
_output_shapes
:@*
dtype0β
conv2d_3/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:????????? @*
strides
*
paddingSAME*
T0²
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? @*
T0d
"batch_normalization_3/LogicalAnd/xConst*
value	B
 Z *
_output_shapes
: *
dtype0
d
"batch_normalization_3/LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
 
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: Ό
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@ΐ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0ή
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0β
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0·
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *K
_output_shapes9
7:????????? @:@:@:@:@:*
U0*
T0*
epsilon%o:`
batch_normalization_3/ConstConst*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: 
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:????????? @Ό
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@Κ
conv2d_4/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:????????? *
paddingSAME*
T0*
strides
²
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
activation/TanhTanhconv2d_4/BiasAdd:output:0*/
_output_shapes
:????????? *
T0λ	
IdentityIdentityactivation/Tanh:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp: : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : :& "
 
_user_specified_nameinputs: 
α
?
6__inference_batch_normalization_1_layer_call_fn_323176

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-321488*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tin	
2*
Tout
2*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321487
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
­
L
0__inference_up_sampling2d_1_layer_call_fn_321361

inputs
identityΔ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352*-
_gradient_op_typePartitionedCall-321358*
Tin
2*J
_output_shapes8
6:4????????????????????????????????????
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
­
L
0__inference_up_sampling2d_2_layer_call_fn_321551

inputs
identityΔ
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-321548**
config_proto

GPU 

CPU2J 8*J
_output_shapes8
6:4????????????????????????????????????*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542*
Tin
2*
Tout
2
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
ϋ

_
C__inference_reshape_layer_call_and_return_conditional_losses_321979

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Q
Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
value
B :*
dtype0*
_output_shapes
: ©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*0
_output_shapes
:?????????*
T0a
IdentityIdentityReshape:output:0*0
_output_shapes
:?????????*
T0"
identityIdentity:output:0*'
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
»
b
F__inference_activation_layer_call_and_return_conditional_losses_322180

inputs
identity`
TanhTanhinputs*A
_output_shapes/
-:+???????????????????????????*
T0j
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:& "
 
_user_specified_nameinputs
έ
ύ
4__inference_batch_normalization_layer_call_fn_323085

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΤ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-321332*
Tin	
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2**
config_proto

GPU 

CPU2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321331
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 

τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321711

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
value	B
 Z *
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*b
_output_shapesP
N:,???????????????????????????:::::*
T0*
U0*
epsilon%o:*
is_training( J
ConstConst*
valueB
 *ΝΜL?*
_output_shapes
: *
dtype0α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
Ζ
D
(__inference_reshape_layer_call_fn_322995

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tout
2*0
_output_shapes
:?????????*
Tin
2*-
_gradient_op_typePartitionedCall-321985*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_321979**
config_proto

GPU 

CPU2J 8i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:?????????*
T0"
identityIdentity:output:0*'
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
ε/

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323345

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 ZN
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
valueB *
dtype0
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
epsilon%o:*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
U0L
Const_2Const*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0Ί
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
valueB
 *  ?Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0Ϋ
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0ξ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΩ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:@«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 Ύ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpΩ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: α
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@ψ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@α
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
:@*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0ϊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
©
ͺ
)__inference_conv2d_1_layer_call_fn_321385

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*-
_gradient_op_typePartitionedCall-321380**
config_proto

GPU 

CPU2J 8*
Tin
2*B
_output_shapes0
.:,???????????????????????????*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Ξ
μ
+__inference_sequential_layer_call_fn_322423
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identity’StatefulPartitionedCallϊ	
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28**
config_proto

GPU 

CPU2J 8*
Tout
2*(
Tin!
2*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_322391*-
_gradient_op_typePartitionedCall-322392*A
_output_shapes/
-:+???????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : :+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : : : : : 


g
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
T0V
ConstConst*
valueB"      *
dtype0*
_output_shapes
:W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:΅
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
half_pixel_centers(*
T0*J
_output_shapes8
6:4????????????????????????????????????
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
‘<
 
__inference__traced_save_323514
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop
savev2_1_const

identity_1’MergeV2Checkpoints’SaveV2’SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_d45ef01004c647c58dfe91f20a9d7dea/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¬
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*Υ
valueΛBΘB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:₯
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B λ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop"/device:CPU:0**
dtypes 
2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B Γ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 Ή
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*
_input_shapes
: :	d::::::::::::::::::::@:@:@:@:@:@:@:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : 
Ώ
η
+__inference_sequential_layer_call_fn_322925

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identity’StatefulPartitionedCallυ	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_322303*
Tout
2*-
_gradient_op_typePartitionedCall-322304*(
Tin!
2*A
_output_shapes/
-:+???????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 

τ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321901

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0²
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Ά
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( *
U0J
ConstConst*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0ΰ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 

τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323167

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( J
ConstConst*
valueB
 *ΝΜL?*
_output_shapes
: *
dtype0α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
a

F__inference_sequential_layer_call_and_return_conditional_losses_322248
dense_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_321948*(
_output_shapes
:?????????*
Tout
2*-
_gradient_op_typePartitionedCall-321954Κ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-321985*0
_output_shapes
:?????????*
Tout
2*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_321979**
config_proto

GPU 

CPU2J 8β
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-321168*
Tout
2**
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162*B
_output_shapes0
.:,???????????????????????????*
Tin
2Ί
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_321184*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321190Ω
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321331*
Tout
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321332*B
_output_shapes0
.:,???????????????????????????ς
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tin
2*-
_gradient_op_typePartitionedCall-322030*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024*
Tout
2κ
up_sampling2d_1/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tout
2*T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352*
Tin
2*-
_gradient_op_typePartitionedCall-321358Δ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321380*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374*
Tin
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2η
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321521*
Tout
2*
Tin	
2*-
_gradient_op_typePartitionedCall-321522ψ
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069*-
_gradient_op_typePartitionedCall-322075*
Tin
2*
Tout
2μ
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2*-
_gradient_op_typePartitionedCall-321548**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542Δ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tout
2*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321570**
config_proto

GPU 

CPU2J 8*
Tin
2η
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-321712*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tin	
2*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321711*
Tout
2ψ
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322120*
Tin
2*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114*B
_output_shapes0
.:,???????????????????????????*
Tout
2μ
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732*
Tin
2*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321738Γ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-321760*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754*
Tin
2*
Tout
2ζ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*A
_output_shapes/
-:+???????????????????????????@*
Tout
2*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321901*-
_gradient_op_typePartitionedCall-321902χ
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-322165*
Tout
2*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*
Tin
2Α
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321926*
Tout
2*
Tin
2*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920*A
_output_shapes/
-:+???????????????????????????δ
activation/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*A
_output_shapes/
-:+???????????????????????????*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_322180*
Tout
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322186*
Tin
2
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall: : : : : :+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : : : : : : : : : : : : : 
Ώ
η
+__inference_sequential_layer_call_fn_322958

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identity’StatefulPartitionedCallυ	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*-
_gradient_op_typePartitionedCall-322392*(
Tin!
2**
config_proto

GPU 

CPU2J 8*
Tout
2*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_322391*A
_output_shapes/
-:+???????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
Τ
§
&__inference_dense_layer_call_fn_322976

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-321954*
Tout
2*
Tin
2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_321948**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:?????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
ψ/

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321487

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
valueB *
dtype0
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*b
_output_shapesP
N:,???????????????????????????:::::*
U0*
epsilon%o:*
T0L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL?»
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
: ά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:*
T0Ϊ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:*
T0«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 *
dtype0Ώ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
valueB
 *  ?Ω
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs: : : : 
­
L
0__inference_up_sampling2d_3_layer_call_fn_321741

inputs
identityΔ
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4????????????????????????????????????*
Tout
2*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732*
Tin
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321738
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
Σ	
Ϊ
A__inference_dense_layer_call_and_return_conditional_losses_322969

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	d*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????‘
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:?????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

J
.__inference_leaky_re_lu_1_layer_call_fn_323195

inputs
identityΊ
PartitionedCallPartitionedCallinputs*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069*
Tin
2*
Tout
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322075*B
_output_shapes0
.:,???????????????????????????{
IdentityIdentityPartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs

J
.__inference_leaky_re_lu_2_layer_call_fn_323295

inputs
identityΊ
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-322120**
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114*B
_output_shapes0
.:,???????????????????????????*
Tin
2*
Tout
2{
IdentityIdentityPartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs


e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
_output_shapes
:V
ConstConst*
valueB"      *
_output_shapes
:*
dtype0W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:΅
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*J
_output_shapes8
6:4????????????????????????????????????*
T0*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
σm

"__inference__traced_restore_323611
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias$
 assignvariableop_2_conv2d_kernel"
assignvariableop_3_conv2d_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance&
"assignvariableop_8_conv2d_1_kernel$
 assignvariableop_9_conv2d_1_bias3
/assignvariableop_10_batch_normalization_1_gamma2
.assignvariableop_11_batch_normalization_1_beta9
5assignvariableop_12_batch_normalization_1_moving_mean=
9assignvariableop_13_batch_normalization_1_moving_variance'
#assignvariableop_14_conv2d_2_kernel%
!assignvariableop_15_conv2d_2_bias3
/assignvariableop_16_batch_normalization_2_gamma2
.assignvariableop_17_batch_normalization_2_beta9
5assignvariableop_18_batch_normalization_2_moving_mean=
9assignvariableop_19_batch_normalization_2_moving_variance'
#assignvariableop_20_conv2d_3_kernel%
!assignvariableop_21_conv2d_3_bias3
/assignvariableop_22_batch_normalization_3_gamma2
.assignvariableop_23_batch_normalization_3_beta9
5assignvariableop_24_batch_normalization_3_moving_mean=
9assignvariableop_25_batch_normalization_3_moving_variance'
#assignvariableop_26_conv2d_4_kernel%
!assignvariableop_27_conv2d_4_bias
identity_29’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9’	RestoreV2’RestoreV2_1―
RestoreV2/tensor_namesConst"/device:CPU:0*Υ
valueΛBΘB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0~
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_1_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_2_kernelIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_2_biasIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_3_kernelIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_3_biasIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_3_gammaIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_3_betaIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_3_moving_meanIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_3_moving_varianceIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2d_4_kernelIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv2d_4_biasIdentity_27:output:0*
_output_shapes
 *
dtype0
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0΅
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ·
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: Δ
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_27:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 


g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
_output_shapes
:*
T0V
ConstConst*
_output_shapes
:*
valueB"      *
dtype0W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:΅
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
half_pixel_centers(*
T0*J
_output_shapes8
6:4????????????????????????????????????
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs

τ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323367

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@²
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Ά
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
is_training( *
U0*
epsilon%o:J
ConstConst*
valueB
 *ΝΜL?*
_output_shapes
: *
dtype0ΰ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 


έ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp¬
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,???????????????????????????*
strides
*
T0*
paddingSAME‘
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????€
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321331

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%o:*
T0*b
_output_shapesP
N:,???????????????????????????:::::*
U0J
ConstConst*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
φ/

O__inference_batch_normalization_layer_call_and_return_conditional_losses_321297

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 ZN
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:H
ConstConst*
dtype0*
valueB *
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::L
Const_2Const*
valueB
 *ΝΜL?*
_output_shapes
: *
dtype0»
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *  ?*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΣ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes	
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΪ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 Ώ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
valueB
 *  ?*
dtype0Ω
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0β
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes	
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
©
ͺ
)__inference_conv2d_2_layer_call_fn_321575

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-321570*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*B
_output_shapes0
.:,???????????????????????????*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
α
?
6__inference_batch_normalization_2_layer_call_fn_323285

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-321712**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*
Tin	
2*
Tout
2*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321711
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
Ρ
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_323290

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs
ή
?
6__inference_batch_normalization_3_layer_call_fn_323376

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΥ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*A
_output_shapes/
-:+???????????????????????????@*
Tout
2*-
_gradient_op_typePartitionedCall-321868*
Tin	
2**
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321867
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
Ρ
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs
a

F__inference_sequential_layer_call_and_return_conditional_losses_322194
dense_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_321948*
Tout
2*-
_gradient_op_typePartitionedCall-321954**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:?????????Κ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-321985*0
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*
Tout
2*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_321979β
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-321168*R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tout
2Ί
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-321190**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_321184Ω
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin	
2*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321297*-
_gradient_op_typePartitionedCall-321298ς
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-322030*
Tin
2*B
_output_shapes0
.:,???????????????????????????*
Tout
2*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024**
config_proto

GPU 

CPU2J 8κ
up_sampling2d_1/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-321358*
Tin
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352*B
_output_shapes0
.:,???????????????????????????*
Tout
2Δ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*-
_gradient_op_typePartitionedCall-321380**
config_proto

GPU 

CPU2J 8*
Tout
2*B
_output_shapes0
.:,???????????????????????????*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374η
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321487*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321488*
Tout
2*
Tin	
2ψ
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-322075**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*B
_output_shapes0
.:,???????????????????????????*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069μ
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*-
_gradient_op_typePartitionedCall-321548*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542*
Tout
2Δ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*B
_output_shapes0
.:,???????????????????????????*
Tout
2*
Tin
2*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321570η
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321678**
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321677*
Tin	
2*
Tout
2ψ
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-322120**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114*
Tout
2*
Tin
2μ
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732*B
_output_shapes0
.:,???????????????????????????*
Tout
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321738Γ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-321760*
Tout
2**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????@*
Tin
2*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754ζ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321868*A
_output_shapes/
-:+???????????????????????????@*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321867χ
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322165*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159Α
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920*A
_output_shapes/
-:+???????????????????????????*-
_gradient_op_typePartitionedCall-321926δ
activation/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????*
Tin
2*-
_gradient_op_typePartitionedCall-322186*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_322180*
Tout
2
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall:+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
Σ	
Ϊ
A__inference_dense_layer_call_and_return_conditional_losses_321948

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	dj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:?????????*
T0‘
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
©
J
.__inference_up_sampling2d_layer_call_fn_321171

inputs
identityΒ
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????*-
_gradient_op_typePartitionedCall-321168
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
₯
¨
'__inference_conv2d_layer_call_fn_321195

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*B
_output_shapes0
.:,???????????????????????????*
Tin
2*-
_gradient_op_typePartitionedCall-321190*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_321184**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
β

!__inference__wrapped_model_321147
dense_input3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource<
8sequential_batch_normalization_3_readvariableop_resource>
:sequential_batch_normalization_3_readvariableop_1_resourceM
Isequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_4_conv2d_readvariableop_resource7
3sequential_conv2d_4_biasadd_readvariableop_resource
identity’>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’-sequential/batch_normalization/ReadVariableOp’/sequential/batch_normalization/ReadVariableOp_1’@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_1/ReadVariableOp’1sequential/batch_normalization_1/ReadVariableOp_1’@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_2/ReadVariableOp’1sequential/batch_normalization_2/ReadVariableOp_1’@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_3/ReadVariableOp’1sequential/batch_normalization_3/ReadVariableOp_1’(sequential/conv2d/BiasAdd/ReadVariableOp’'sequential/conv2d/Conv2D/ReadVariableOp’*sequential/conv2d_1/BiasAdd/ReadVariableOp’)sequential/conv2d_1/Conv2D/ReadVariableOp’*sequential/conv2d_2/BiasAdd/ReadVariableOp’)sequential/conv2d_2/Conv2D/ReadVariableOp’*sequential/conv2d_3/BiasAdd/ReadVariableOp’)sequential/conv2d_3/Conv2D/ReadVariableOp’*sequential/conv2d_4/BiasAdd/ReadVariableOp’)sequential/conv2d_4/Conv2D/ReadVariableOp’'sequential/dense/BiasAdd/ReadVariableOp’&sequential/dense/MatMul/ReadVariableOpΕ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	d
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*(
_output_shapes
:?????????*
T0Γ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ͺ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????k
sequential/reshape/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:p
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0r
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0r
(sequential/reshape/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:°
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
_output_shapes
: *
Index0d
"sequential/reshape/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: d
"sequential/reshape/Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: e
"sequential/reshape/Reshape/shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0+sequential/reshape/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:°
sequential/reshape/ReshapeReshape#sequential/dense/Relu:activations:0)sequential/reshape/Reshape/shape:output:0*0
_output_shapes
:?????????*
T0q
sequential/up_sampling2d/ShapeShape#sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,sequential/up_sampling2d/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:x
.sequential/up_sampling2d/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:x
.sequential/up_sampling2d/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Ί
&sequential/up_sampling2d/strided_sliceStridedSlice'sequential/up_sampling2d/Shape:output:05sequential/up_sampling2d/strided_slice/stack:output:07sequential/up_sampling2d/strided_slice/stack_1:output:07sequential/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
_output_shapes
:*
T0o
sequential/up_sampling2d/ConstConst*
dtype0*
valueB"      *
_output_shapes
:’
sequential/up_sampling2d/mulMul/sequential/up_sampling2d/strided_slice:output:0'sequential/up_sampling2d/Const:output:0*
_output_shapes
:*
T0κ
5sequential/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#sequential/reshape/Reshape:output:0 sequential/up_sampling2d/mul:z:0*0
_output_shapes
:?????????*
T0*
half_pixel_centers(Π
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0ώ
sequential/conv2d/Conv2DConv2DFsequential/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*
strides
*0
_output_shapes
:?????????Ε
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:΄
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
+sequential/batch_normalization/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: m
+sequential/batch_normalization/LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: »
)sequential/batch_normalization/LogicalAnd
LogicalAnd4sequential/batch_normalization/LogicalAnd/x:output:04sequential/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: Ο
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Σ
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ρ
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:υ
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ς
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
is_training( *
epsilon%o:*P
_output_shapes>
<:?????????:::::*
U0i
$sequential/batch_normalization/ConstConst*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*0
_output_shapes
:?????????~
 sequential/up_sampling2d_1/ShapeShape.sequential/leaky_re_lu/LeakyRelu:activations:0*
_output_shapes
:*
T0x
.sequential/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0sequential/up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:z
0sequential/up_sampling2d_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Δ
(sequential/up_sampling2d_1/strided_sliceStridedSlice)sequential/up_sampling2d_1/Shape:output:07sequential/up_sampling2d_1/strided_slice/stack:output:09sequential/up_sampling2d_1/strided_slice/stack_1:output:09sequential/up_sampling2d_1/strided_slice/stack_2:output:0*
T0*
Index0*
_output_shapes
:q
 sequential/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      ¨
sequential/up_sampling2d_1/mulMul1sequential/up_sampling2d_1/strided_slice:output:0)sequential/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:ω
7sequential/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor.sequential/leaky_re_lu/LeakyRelu:activations:0"sequential/up_sampling2d_1/mul:z:0*0
_output_shapes
:?????????*
T0*
half_pixel_centers(Τ
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:*
dtype0
sequential/conv2d_1/Conv2DConv2DHsequential/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
strides
*0
_output_shapes
:?????????*
T0*
paddingSAMEΙ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ί
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:?????????*
T0o
-sequential/batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
o
-sequential/batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
Α
+sequential/batch_normalization_1/LogicalAnd
LogicalAnd6sequential/batch_normalization_1/LogicalAnd/x:output:06sequential/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: Σ
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Χ
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:υ
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ω
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ώ
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d_1/BiasAdd:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*P
_output_shapes>
<:?????????:::::*
is_training( *
T0*
U0*
epsilon%o:k
&sequential/batch_normalization_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *ΝΜL?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????
 sequential/up_sampling2d_2/ShapeShape0sequential/leaky_re_lu_1/LeakyRelu:activations:0*
_output_shapes
:*
T0x
.sequential/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0sequential/up_sampling2d_2/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0z
0sequential/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Δ
(sequential/up_sampling2d_2/strided_sliceStridedSlice)sequential/up_sampling2d_2/Shape:output:07sequential/up_sampling2d_2/strided_slice/stack:output:09sequential/up_sampling2d_2/strided_slice/stack_1:output:09sequential/up_sampling2d_2/strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
T0q
 sequential/up_sampling2d_2/ConstConst*
_output_shapes
:*
valueB"      *
dtype0¨
sequential/up_sampling2d_2/mulMul1sequential/up_sampling2d_2/strided_slice:output:0)sequential/up_sampling2d_2/Const:output:0*
_output_shapes
:*
T0ϋ
7sequential/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor0sequential/leaky_re_lu_1/LeakyRelu:activations:0"sequential/up_sampling2d_2/mul:z:0*0
_output_shapes
:?????????*
T0*
half_pixel_centers(Τ
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:
sequential/conv2d_2/Conv2DConv2DHsequential/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
paddingSAME*
T0*0
_output_shapes
:?????????*
strides
Ι
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Ί
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????o
-sequential/batch_normalization_2/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: o
-sequential/batch_normalization_2/LogicalAnd/yConst*
value	B
 Z*
_output_shapes
: *
dtype0
Α
+sequential/batch_normalization_2/LogicalAnd
LogicalAnd6sequential/batch_normalization_2/LogicalAnd/x:output:06sequential/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: Σ
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Χ
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0υ
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ω
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ώ
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d_2/BiasAdd:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*P
_output_shapes>
<:?????????:::::*
T0*
U0*
epsilon%o:*
is_training( k
&sequential/batch_normalization_2/ConstConst*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0
"sequential/leaky_re_lu_2/LeakyRelu	LeakyRelu5sequential/batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????
 sequential/up_sampling2d_3/ShapeShape0sequential/leaky_re_lu_2/LeakyRelu:activations:0*
_output_shapes
:*
T0x
.sequential/up_sampling2d_3/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:z
0sequential/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential/up_sampling2d_3/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0Δ
(sequential/up_sampling2d_3/strided_sliceStridedSlice)sequential/up_sampling2d_3/Shape:output:07sequential/up_sampling2d_3/strided_slice/stack:output:09sequential/up_sampling2d_3/strided_slice/stack_1:output:09sequential/up_sampling2d_3/strided_slice/stack_2:output:0*
T0*
_output_shapes
:*
Index0q
 sequential/up_sampling2d_3/ConstConst*
valueB"      *
dtype0*
_output_shapes
:¨
sequential/up_sampling2d_3/mulMul1sequential/up_sampling2d_3/strided_slice:output:0)sequential/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:ϋ
7sequential/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor0sequential/leaky_re_lu_2/LeakyRelu:activations:0"sequential/up_sampling2d_3/mul:z:0*
half_pixel_centers(*
T0*0
_output_shapes
:????????? Σ
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*'
_output_shapes
:@*
dtype0
sequential/conv2d_3/Conv2DConv2DHsequential/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*
paddingSAME*
strides
*/
_output_shapes
:????????? @Θ
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ή
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? @*
T0o
-sequential/batch_normalization_3/LogicalAnd/xConst*
dtype0
*
value	B
 Z *
_output_shapes
: o
-sequential/batch_normalization_3/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 ZΑ
+sequential/batch_normalization_3/LogicalAnd
LogicalAnd6sequential/batch_normalization_3/LogicalAnd/x:output:06sequential/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: ?
/sequential/batch_normalization_3/ReadVariableOpReadVariableOp8sequential_batch_normalization_3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Φ
1sequential/batch_normalization_3/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0τ
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@ψ
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0ω
1sequential/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d_3/BiasAdd:output:07sequential/batch_normalization_3/ReadVariableOp:value:09sequential/batch_normalization_3/ReadVariableOp_1:value:0Hsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
epsilon%o:*
T0*K
_output_shapes9
7:????????? @:@:@:@:@:*
is_training( k
&sequential/batch_normalization_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *ΝΜL?
"sequential/leaky_re_lu_3/LeakyRelu	LeakyRelu5sequential/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:????????? @?
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@*
dtype0λ
sequential/conv2d_4/Conv2DConv2D0sequential/leaky_re_lu_3/LeakyRelu:activations:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*
paddingSAME*/
_output_shapes
:????????? *
strides
Θ
*sequential/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ή
sequential/conv2d_4/BiasAddBiasAdd#sequential/conv2d_4/Conv2D:output:02sequential/conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0
sequential/activation/TanhTanh$sequential/conv2d_4/BiasAdd:output:0*/
_output_shapes
:????????? *
T0ͺ
IdentityIdentitysequential/activation/Tanh:y:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1A^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_3/ReadVariableOp2^sequential/batch_normalization_3/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp+^sequential/conv2d_4/BiasAdd/ReadVariableOp*^sequential/conv2d_4/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*/
_output_shapes
:????????? *
T0"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2f
1sequential/batch_normalization_3/ReadVariableOp_11sequential/batch_normalization_3/ReadVariableOp_12R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12V
)sequential/conv2d_4/Conv2D/ReadVariableOp)sequential/conv2d_4/Conv2D/ReadVariableOp2b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2b
/sequential/batch_normalization_3/ReadVariableOp/sequential/batch_normalization_3/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12X
*sequential/conv2d_4/BiasAdd/ReadVariableOp*sequential/conv2d_4/BiasAdd/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp: : : : : : : : : : : : : : : :+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : : : 
Ρ
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs
α
?
6__inference_batch_normalization_2_layer_call_fn_323276

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*-
_gradient_op_typePartitionedCall-321678*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321677*
Tin	
2*B
_output_shapes0
.:,???????????????????????????**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 

G
+__inference_activation_layer_call_fn_323405

inputs
identityΆ
PartitionedCallPartitionedCallinputs*
Tin
2**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_322180*
Tout
2*-
_gradient_op_typePartitionedCall-322186z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:& "
 
_user_specified_nameinputs
Ο
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_323090

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs


g
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
dtype0*
_output_shapes
:*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
_output_shapes
:*
T0΅
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
half_pixel_centers(*J
_output_shapes8
6:4????????????????????????????????????*
T0
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
έ
ύ
4__inference_batch_normalization_layer_call_fn_323076

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΤ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*B
_output_shapes0
.:,???????????????????????????*
Tout
2*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321297*
Tin	
2*-
_gradient_op_typePartitionedCall-321298**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
ή
?
6__inference_batch_normalization_3_layer_call_fn_323385

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity’StatefulPartitionedCallΥ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321901*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*-
_gradient_op_typePartitionedCall-321902
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
ψ/

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323245

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
value	B
 Z*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0H
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
_output_shapes
: *
dtype0*
valueB 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*b
_output_shapesP
N:,???????????????????????????:::::*
T0*
U0L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL?»
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
valueB
 *  ?*
_output_shapes
: *
dtype0Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0Ϊ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes	
:«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ώ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
dtype0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpΩ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:β
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
 ϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
a

F__inference_sequential_layer_call_and_return_conditional_losses_322391

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_4+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_4+
'conv2d_4_statefulpartitionedcall_args_1+
'conv2d_4_statefulpartitionedcall_args_2
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCallό
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_321948*(
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-321954*
Tout
2**
config_proto

GPU 

CPU2J 8Κ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????*
Tin
2*-
_gradient_op_typePartitionedCall-321985*
Tout
2*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_321979**
config_proto

GPU 

CPU2J 8β
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*-
_gradient_op_typePartitionedCall-321168*B
_output_shapes0
.:,???????????????????????????*R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162*
Tout
2Ί
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321190**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_321184*
Tin
2*
Tout
2Ω
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin	
2*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_321331*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321332ς
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????*P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024**
config_proto

GPU 

CPU2J 8*
Tin
2*-
_gradient_op_typePartitionedCall-322030*
Tout
2κ
up_sampling2d_1/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*-
_gradient_op_typePartitionedCall-321358*
Tin
2*B
_output_shapes0
.:,???????????????????????????*T
fORM
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352**
config_proto

GPU 

CPU2J 8*
Tout
2Δ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*B
_output_shapes0
.:,???????????????????????????*
Tin
2*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374**
config_proto

GPU 

CPU2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-321380η
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-321522*
Tin	
2**
config_proto

GPU 

CPU2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321521*
Tout
2*B
_output_shapes0
.:,???????????????????????????ψ
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tout
2*-
_gradient_op_typePartitionedCall-322075*R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_322069*
Tin
2**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,???????????????????????????μ
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321548*B
_output_shapes0
.:,???????????????????????????*
Tout
2*
Tin
2*T
fORM
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542Δ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321570*
Tout
2**
config_proto

GPU 

CPU2J 8η
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321712*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321711*
Tout
2ψ
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-322120*B
_output_shapes0
.:,???????????????????????????*R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_322114*
Tout
2*
Tin
2μ
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*B
_output_shapes0
.:,???????????????????????????*-
_gradient_op_typePartitionedCall-321738*T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8Γ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+???????????????????????????@*-
_gradient_op_typePartitionedCall-321760**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754*
Tout
2ζ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*A
_output_shapes/
-:+???????????????????????????@*
Tout
2*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321901*
Tin	
2**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321902χ
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159*-
_gradient_op_typePartitionedCall-322165**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????@Α
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'conv2d_4_statefulpartitionedcall_args_1'conv2d_4_statefulpartitionedcall_args_2*M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920**
config_proto

GPU 

CPU2J 8*-
_gradient_op_typePartitionedCall-321926*
Tin
2*A
_output_shapes/
-:+???????????????????????????*
Tout
2δ
activation/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_322180*
Tin
2*-
_gradient_op_typePartitionedCall-322186*
Tout
2**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+???????????????????????????
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall: : : : : : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
Ο
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_322024

inputs
identityb
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,???????????????????????????z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*A
_input_shapes0
.:,???????????????????????????:& "
 
_user_specified_nameinputs


έ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp«
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:@«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+???????????????????????????@*
T0*
strides
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????@*
T0£
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_321521

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
U0*
epsilon%o:*b
_output_shapesP
N:,???????????????????????????:::::*
T0J
ConstConst*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
Ξ
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_322159

inputs
identitya
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityLeakyRelu:activations:0*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:& "
 
_user_specified_nameinputs
ψ/

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_321677

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
valueB *
_output_shapes
: 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL?»
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0ά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes	
:Ϊ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes	
:*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΏ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
valueB
 *  ?*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpΩ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: β
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
T0*
_output_shapes	
:β
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:*
T0΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0ϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
φ/

O__inference_batch_normalization_layer_call_and_return_conditional_losses_323045

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0H
ConstConst*
valueB *
_output_shapes
: *
dtype0J
Const_1Const*
_output_shapes
: *
dtype0*
valueB 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
T0L
Const_2Const*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: »
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
valueB
 *  ?Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpά
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ο
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΪ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpΏ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpΩ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ω
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes	
:*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpβ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ϋ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,???????????????????????????"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
ε/

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_321867

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’#AssignMovingAvg/Read/ReadVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’%AssignMovingAvg_1/Read/ReadVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0H
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
valueB *
_output_shapes
: *
dtype0
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%o:*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:L
Const_2Const*
_output_shapes
: *
valueB
 *ΝΜL?*
dtype0Ί
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0ΐ
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
valueB
 *  ?Σ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Ϋ
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0ξ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ω
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
T0*
_output_shapes
:@«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
 Ύ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Δ
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  ?*
dtype0*
_output_shapes
: *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpΩ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: α
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0ψ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@α
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@΅
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0ϊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
»
b
F__inference_activation_layer_call_and_return_conditional_losses_323400

inputs
identity`
TanhTanhinputs*A
_output_shapes/
-:+???????????????????????????*
T0j
IdentityIdentityTanh:y:0*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????:& "
 
_user_specified_nameinputs
Ξ
μ
+__inference_sequential_layer_call_fn_322335
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identity’StatefulPartitionedCallϊ	
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*-
_gradient_op_typePartitionedCall-322304*A
_output_shapes/
-:+???????????????????????????*(
Tin!
2**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_322303*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : : : : : :+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : 
ώ
ε
$__inference_signature_wrapper_322514
dense_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28
identity’StatefulPartitionedCallΓ	
StatefulPartitionedCallStatefulPartitionedCalldense_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28*-
_gradient_op_typePartitionedCall-322483*(
Tin!
2**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_321147*/
_output_shapes
:????????? *
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*
_input_shapes
:?????????d::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_namedense_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
ϋ

_
C__inference_reshape_layer_call_and_return_conditional_losses_322990

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:Ρ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
value
B :*
dtype0©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????"
identityIdentity:output:0*'
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
Ξ
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_323390

inputs
identitya
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:& "
 
_user_specified_nameinputs

ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_323067

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0³
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0·
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%o:*b
_output_shapesP
N:,???????????????????????????:::::*
U0*
T0J
ConstConst*
valueB
 *ΝΜL?*
dtype0*
_output_shapes
: α
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,???????????????????????????*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 


έ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOpͺ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+???????????????????????????*
strides
*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0£
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*½
serving_default©
C
dense_input4
serving_default_dense_input:0?????????dF

activation8
StatefulPartitionedCall:0????????? tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Μ
Ηv
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
	variables
regularization_losses
trainable_variables
	keras_api

signatures
η_default_save_signature
θ__call__
+ι&call_and_return_all_conditional_losses"θp
_tf_keras_sequentialΙp{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [2, 3, 256]}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": [2, 1], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [2, 3, 256]}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": [2, 1], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}}}
―
regularization_losses
	variables
trainable_variables
	keras_api
κ__call__
+λ&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 100], "config": {"batch_input_shape": [null, 100], "dtype": "float32", "sparse": false, "name": "dense_input"}}


kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
μ__call__
+ν&call_and_return_all_conditional_losses"τ
_tf_keras_layerΪ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100], "config": {"name": "dense", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}

%regularization_losses
&	variables
'trainable_variables
(	keras_api
ξ__call__
+ο&call_and_return_all_conditional_losses"
_tf_keras_layerπ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [2, 3, 256]}}
λ
)regularization_losses
*	variables
+trainable_variables
,	keras_api
π__call__
+ρ&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": [2, 1], "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ξ

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
ς__call__
+σ&call_and_return_all_conditional_losses"Η
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
±
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
τ__call__
+υ&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
¦
<regularization_losses
=	variables
>trainable_variables
?	keras_api
φ__call__
+χ&call_and_return_all_conditional_losses"
_tf_keras_layerϋ{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ο
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
ψ__call__
+ω&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ς

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
ϊ__call__
+ϋ&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
΅
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
ό__call__
+ύ&call_and_return_all_conditional_losses"ί
_tf_keras_layerΕ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
ͺ
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
ώ__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ο
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
__call__
+&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ς

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
__call__
+&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
΅
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
fregularization_losses
g	variables
htrainable_variables
i	keras_api
__call__
+&call_and_return_all_conditional_losses"ί
_tf_keras_layerΕ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
ͺ
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ο
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
__call__
+&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ρ

rkernel
sbias
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
__call__
+&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
΅
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}regularization_losses
~	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
υ
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Θ
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
‘
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerς{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "tanh"}}
ψ
0
 1
-2
.3
44
55
66
77
D8
E9
K10
L11
M12
N13
[14
\15
b16
c17
d18
e19
r20
s21
y22
z23
{24
|25
26
27"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
0
 1
-2
.3
44
55
D6
E7
K8
L9
[10
\11
b12
c13
r14
s15
y16
z17
18
19"
trackable_list_wrapper
Ώ
	variables
regularization_losses
layers
non_trainable_variables
metrics
 layer_regularization_losses
trainable_variables
θ__call__
η_default_save_signature
+ι&call_and_return_all_conditional_losses
'ι"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
regularization_losses
layers
	variables
non_trainable_variables
metrics
 layer_regularization_losses
trainable_variables
κ__call__
+λ&call_and_return_all_conditional_losses
'λ"call_and_return_conditional_losses"
_generic_user_object
:	d2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
‘
!regularization_losses
layers
"	variables
non_trainable_variables
metrics
 layer_regularization_losses
#trainable_variables
μ__call__
+ν&call_and_return_all_conditional_losses
'ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
%regularization_losses
layers
&	variables
non_trainable_variables
metrics
 layer_regularization_losses
'trainable_variables
ξ__call__
+ο&call_and_return_all_conditional_losses
'ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
)regularization_losses
layers
*	variables
 non_trainable_variables
‘metrics
 ’layer_regularization_losses
+trainable_variables
π__call__
+ρ&call_and_return_all_conditional_losses
'ρ"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
‘
/regularization_losses
£layers
0	variables
€non_trainable_variables
₯metrics
 ¦layer_regularization_losses
1trainable_variables
ς__call__
+σ&call_and_return_all_conditional_losses
'σ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&2batch_normalization/gamma
':%2batch_normalization/beta
0:. (2batch_normalization/moving_mean
4:2 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
‘
8regularization_losses
§layers
9	variables
¨non_trainable_variables
©metrics
 ͺlayer_regularization_losses
:trainable_variables
τ__call__
+υ&call_and_return_all_conditional_losses
'υ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
<regularization_losses
«layers
=	variables
¬non_trainable_variables
­metrics
 ?layer_regularization_losses
>trainable_variables
φ__call__
+χ&call_and_return_all_conditional_losses
'χ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
@regularization_losses
―layers
A	variables
°non_trainable_variables
±metrics
 ²layer_regularization_losses
Btrainable_variables
ψ__call__
+ω&call_and_return_all_conditional_losses
'ω"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
‘
Fregularization_losses
³layers
G	variables
΄non_trainable_variables
΅metrics
 Άlayer_regularization_losses
Htrainable_variables
ϊ__call__
+ϋ&call_and_return_all_conditional_losses
'ϋ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_1/gamma
):'2batch_normalization_1/beta
2:0 (2!batch_normalization_1/moving_mean
6:4 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
‘
Oregularization_losses
·layers
P	variables
Έnon_trainable_variables
Ήmetrics
 Ίlayer_regularization_losses
Qtrainable_variables
ό__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
Sregularization_losses
»layers
T	variables
Όnon_trainable_variables
½metrics
 Ύlayer_regularization_losses
Utrainable_variables
ώ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
Wregularization_losses
Ώlayers
X	variables
ΐnon_trainable_variables
Αmetrics
 Βlayer_regularization_losses
Ytrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
‘
]regularization_losses
Γlayers
^	variables
Δnon_trainable_variables
Εmetrics
 Ζlayer_regularization_losses
_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_2/gamma
):'2batch_normalization_2/beta
2:0 (2!batch_normalization_2/moving_mean
6:4 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
‘
fregularization_losses
Ηlayers
g	variables
Θnon_trainable_variables
Ιmetrics
 Κlayer_regularization_losses
htrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
jregularization_losses
Λlayers
k	variables
Μnon_trainable_variables
Νmetrics
 Ξlayer_regularization_losses
ltrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
nregularization_losses
Οlayers
o	variables
Πnon_trainable_variables
Ρmetrics
 ?layer_regularization_losses
ptrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_3/kernel
:@2conv2d_3/bias
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
‘
tregularization_losses
Σlayers
u	variables
Τnon_trainable_variables
Υmetrics
 Φlayer_regularization_losses
vtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
‘
}regularization_losses
Χlayers
~	variables
Ψnon_trainable_variables
Ωmetrics
 Ϊlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
€
regularization_losses
Ϋlayers
	variables
άnon_trainable_variables
έmetrics
 ήlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'@2conv2d_4/kernel
:2conv2d_4/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
€
regularization_losses
ίlayers
	variables
ΰnon_trainable_variables
αmetrics
 βlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
€
regularization_losses
γlayers
	variables
δnon_trainable_variables
εmetrics
 ζlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ά
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
X
60
71
M2
N3
d4
e5
{6
|7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
γ2ΰ
!__inference__wrapped_model_321147Ί
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ **’'
%"
dense_input?????????d
ϊ2χ
+__inference_sequential_layer_call_fn_322423
+__inference_sequential_layer_call_fn_322958
+__inference_sequential_layer_call_fn_322925
+__inference_sequential_layer_call_fn_322335ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ζ2γ
F__inference_sequential_layer_call_and_return_conditional_losses_322194
F__inference_sequential_layer_call_and_return_conditional_losses_322732
F__inference_sequential_layer_call_and_return_conditional_losses_322248
F__inference_sequential_layer_call_and_return_conditional_losses_322892ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Μ2ΙΖ
½²Ή
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 
Μ2ΙΖ
½²Ή
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 
Π2Ν
&__inference_dense_layer_call_fn_322976’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
λ2θ
A__inference_dense_layer_call_and_return_conditional_losses_322969’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2Ο
(__inference_reshape_layer_call_fn_322995’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_reshape_layer_call_and_return_conditional_losses_322990’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
.__inference_up_sampling2d_layer_call_fn_321171ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
±2?
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
'__inference_conv2d_layer_call_fn_321195Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
’2
B__inference_conv2d_layer_call_and_return_conditional_losses_321184Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
¦2£
4__inference_batch_normalization_layer_call_fn_323085
4__inference_batch_normalization_layer_call_fn_323076΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ά2Ω
O__inference_batch_normalization_layer_call_and_return_conditional_losses_323045
O__inference_batch_normalization_layer_call_and_return_conditional_losses_323067΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
,__inference_leaky_re_lu_layer_call_fn_323095’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ρ2ξ
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_323090’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
0__inference_up_sampling2d_1_layer_call_fn_321361ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_conv2d_1_layer_call_fn_321385Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
€2‘
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
ͺ2§
6__inference_batch_normalization_1_layer_call_fn_323176
6__inference_batch_normalization_1_layer_call_fn_323185΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323167
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323145΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
.__inference_leaky_re_lu_1_layer_call_fn_323195’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_323190’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
0__inference_up_sampling2d_2_layer_call_fn_321551ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_conv2d_2_layer_call_fn_321575Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
€2‘
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
ͺ2§
6__inference_batch_normalization_2_layer_call_fn_323276
6__inference_batch_normalization_2_layer_call_fn_323285΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323267
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323245΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
.__inference_leaky_re_lu_2_layer_call_fn_323295’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_323290’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
0__inference_up_sampling2d_3_layer_call_fn_321741ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_conv2d_3_layer_call_fn_321765Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
€2‘
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
ͺ2§
6__inference_batch_normalization_3_layer_call_fn_323376
6__inference_batch_normalization_3_layer_call_fn_323385΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323367
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323345΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
.__inference_leaky_re_lu_3_layer_call_fn_323395’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_323390’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
)__inference_conv2d_4_layer_call_fn_321931Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+???????????????????????????@
£2 
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+???????????????????????????@
Υ2?
+__inference_activation_layer_call_fn_323405’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_activation_layer_call_and_return_conditional_losses_323400’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
7B5
$__inference_signature_wrapper_322514dense_inputά
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_323190J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 Ϋ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_321920I’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+???????????????????????????
 Ϋ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_321374DEJ’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 Ϊ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_321754rsJ’G
@’=
;8
inputs,???????????????????????????
ͺ "?’<
52
0+???????????????????????????@
 μ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3230674567N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 ±
.__inference_leaky_re_lu_3_layer_call_fn_323395I’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+???????????????????????????@³
)__inference_conv2d_2_layer_call_fn_321575[\J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????κ
F__inference_sequential_layer_call_and_return_conditional_losses_322248 -.4567DEKLMN[\bcdersyz{|<’9
2’/
%"
dense_input?????????d
p 

 
ͺ "?’<
52
0+???????????????????????????
 ²
,__inference_leaky_re_lu_layer_call_fn_323095J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????ά
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_323290J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 Δ
4__inference_batch_normalization_layer_call_fn_3230854567N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Β
+__inference_sequential_layer_call_fn_322335 -.4567DEKLMN[\bcdersyz{|<’9
2’/
%"
dense_input?????????d
p

 
ͺ "2/+???????????????????????????³
)__inference_conv2d_4_layer_call_fn_321931I’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+???????????????????????????
(__inference_reshape_layer_call_fn_322995U0’-
&’#
!
inputs?????????
ͺ "!?????????’
A__inference_dense_layer_call_and_return_conditional_losses_322969] /’,
%’"
 
inputs?????????d
ͺ "&’#

0?????????
 ξ
K__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_321542R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ο
$__inference_signature_wrapper_322514¦ -.4567DEKLMN[\bcdersyz{|C’@
’ 
9ͺ6
4
dense_input%"
dense_input?????????d"?ͺ<
:

activation,)

activation????????? Σ
F__inference_sequential_layer_call_and_return_conditional_losses_322732 -.4567DEKLMN[\bcdersyz{|7’4
-’*
 
inputs?????????d
p

 
ͺ "-’*
# 
0????????? 
 Ζ
0__inference_up_sampling2d_2_layer_call_fn_321551R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Ζ
0__inference_up_sampling2d_1_layer_call_fn_321361R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Ζ
0__inference_up_sampling2d_3_layer_call_fn_321741R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Ϊ
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_323090J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 ½
!__inference__wrapped_model_321147 -.4567DEKLMN[\bcdersyz{|4’1
*’'
%"
dense_input?????????d
ͺ "?ͺ<
:

activation,)

activation????????? μ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323367yz{|M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 ΄
.__inference_leaky_re_lu_2_layer_call_fn_323295J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????©
C__inference_reshape_layer_call_and_return_conditional_losses_322990b0’-
&’#
!
inputs?????????
ͺ ".’+
$!
0?????????
 Ζ
6__inference_batch_normalization_1_layer_call_fn_323176KLMNN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Ζ
6__inference_batch_normalization_2_layer_call_fn_323276bcdeN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????μ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3230454567N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 ±
'__inference_conv2d_layer_call_fn_321195-.J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,????????????????????????????
+__inference_activation_layer_call_fn_323405I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????ξ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323245bcdeN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 Ζ
6__inference_batch_normalization_1_layer_call_fn_323185KLMNN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Ω
B__inference_conv2d_layer_call_and_return_conditional_losses_321184-.J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 z
&__inference_dense_layer_call_fn_322976P /’,
%’"
 
inputs?????????d
ͺ "?????????½
+__inference_sequential_layer_call_fn_322958 -.4567DEKLMN[\bcdersyz{|7’4
-’*
 
inputs?????????d
p 

 
ͺ "2/+???????????????????????????΄
.__inference_leaky_re_lu_1_layer_call_fn_323195J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????μ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_323345yz{|M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ξ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323145KLMNN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 Ϊ
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_323390I’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+???????????????????????????@
 ½
+__inference_sequential_layer_call_fn_322925 -.4567DEKLMN[\bcdersyz{|7’4
-’*
 
inputs?????????d
p

 
ͺ "2/+???????????????????????????ξ
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_321732R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ³
)__inference_conv2d_1_layer_call_fn_321385DEJ’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????Δ
4__inference_batch_normalization_layer_call_fn_3230764567N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????μ
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_321162R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
6__inference_batch_normalization_2_layer_call_fn_323285bcdeN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????²
)__inference_conv2d_3_layer_call_fn_321765rsJ’G
@’=
;8
inputs,???????????????????????????
ͺ "2/+???????????????????????????@ξ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_323167KLMNN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 ξ
K__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_321352R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ξ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_323267bcdeN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 Δ
.__inference_up_sampling2d_layer_call_fn_321171R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Δ
6__inference_batch_normalization_3_layer_call_fn_323385yz{|M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@Ϋ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_321564[\J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 Σ
F__inference_sequential_layer_call_and_return_conditional_losses_322892 -.4567DEKLMN[\bcdersyz{|7’4
-’*
 
inputs?????????d
p 

 
ͺ "-’*
# 
0????????? 
 Δ
6__inference_batch_normalization_3_layer_call_fn_323376yz{|M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Χ
F__inference_activation_layer_call_and_return_conditional_losses_323400I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????
 Β
+__inference_sequential_layer_call_fn_322423 -.4567DEKLMN[\bcdersyz{|<’9
2’/
%"
dense_input?????????d
p 

 
ͺ "2/+???????????????????????????κ
F__inference_sequential_layer_call_and_return_conditional_losses_322194 -.4567DEKLMN[\bcdersyz{|<’9
2’/
%"
dense_input?????????d
p

 
ͺ "?’<
52
0+???????????????????????????
 