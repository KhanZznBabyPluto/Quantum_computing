\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{msukz}[2016/02/04 Thesis LaTeX class]
\LoadClass[a4paper, 12pt]{extbook}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[T2A]{fontenc}		% �������� pdf-����� � ���������� ���������� �������� ������
\usepackage[cp1251]{inputenc}		% �������� tex-����� ���������� � ����������� WINDOWS ��������� �p-1251
\usepackage[english, russian]{babel}	% �������� ������
%\usepackage{cmap}				% ����������� ������ �� pdf

\usepackage{amssymb}			% amsfonts + �������������� �������������� ��������
\usepackage{amsthm}				% �������  
\usepackage{amsmath}			% ������������� �������
\usepackage{textcomp}			% ������� ����� ������������ ������� \No

\usepackage[left=2.5cm, right=2.5cm, top=2cm, bottom=2cm]{geometry} % �������
\parindent=0.75cm				% ������ ���������
\binoppenalty=10000				% ��������� ������� ������ �� ����� ������ �� ��������� ��������� + - * =
\relpenalty=10000				% ��������� ������� ������ �� ����� ������ �� ��������� ��������� > <
\sloppy							% �������� ��������� � ������������ �����, �������� ������ ����� �� ������ ������.

\usepackage{enumitem}
\setlist{leftmargin=\parindent}% ��������� ������� ��������� ������

\usepackage{fancyhdr}			
\pagestyle{fancy}				% ��������� ������������
\fancyhf{}						% ������ ������
\fancyfoot[C]{\thepage}			% ����� ����� �������� �� ������
\renewcommand{\headrulewidth}{0pt} % ������ �������������� �����

\usepackage{listings}			% ��� ��������
\lstset{
	numbers=left, 
	numbersep=5pt, 
    frame=single,
    basicstyle=\ttfamily
}

\theoremstyle{plain}
\newtheorem{theorem}{�������}
\newtheorem{lemma}{�����}
\newtheorem{corollary}{���������}
\newtheorem{definition}{�����������}

\usepackage{graphicx}			% �����������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����� ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% #1 - �������
% #2 - ����
% #3 - ���������
% #4 - �����������
% #5 - �����, ������
% #6 - email

\newcommand{\info}[6]
{
	\begin{center}
	\textbf{#2} \\
	\textit{\textbf{#1}} \\
	\textit{#3} \\
	\textit{#4} \\
	\textit{#5} \\
	\textit{\tt #6} \nopagebreak
	\end{center}

	\addcontentsline{toc}{subsection}{\textit{#1} #2}
	\setcounter{equation}{0}
	\setcounter{theorem}{0}
	\setcounter{figure}{0}
	\setcounter{theorem}{0}
	\setcounter{lemma}{0}
}

% ������
% \thesis
% {������~�.�., ������~�.�.}
% {������ ���������� ������ � ���������, ��������� � ���������}
% {����������, �������}
% {������������� ������ ��� ����� �.�.����������}
% {�.~������, ���������}
% {ivanov@msu.kz, petrov@msu.kz}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �����������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% #1 - ��� ����� � ������������ (��� ����������)
% #2 - �������
\newcommand{\onepicture}[2]
{
	\begin{figure}[h!]
	\begin{center}
		\includegraphics[width=400pt]{#1}
		\caption{#2}
	\end{center}
	\end{figure}
}

% #1 - ��� 1-�� ����� � ������������ (��� ����������)
% #2 - ������� 1-�� �����������
% #1 - ��� 2-�� ����� � ������������ (��� ����������)
% #2 - ������� 2-�� �����������

\newcommand{\twopicture}[4]
{
	\begin{figure}[h!]
	\begin{center}
		\begin{minipage}{0.45\linewidth}
		\begin{center}
			\includegraphics[width=200pt]{#1}
			\caption{#2}
		\end{center}
		\end{minipage}
	\hfill 
		\begin{minipage}{0.45\linewidth}
		\begin{center}
			\includegraphics[width=200pt]{#3}
			\caption{#4}
		\end{center}
		\end{minipage}
	\end{center}
	\end{figure}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������ ����������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{references}
{
	\nopagebreak
	~\\
	\textbf{����������������� ������}
	\begin{enumerate}
}{
	\end{enumerate}
}
% �����
% #1 - ������
% #2 - ��������
% #3 - �����
% #4 - ������������
% #5 - ���
% #6 - �������
\newcommand{\bookRu}[6]
{
	\item #1 #2. #3: #4, #5. #6~�.
}
\newcommand{\bookEn}[6]
{
	\selectlanguage{english}
	\item #1 #2. #3: #4, #5. #6~p.
	\selectlanguage{russian}
}
\newcommand{\bookKz}[6]
{
	\item #1 #2. #3: #4, #5. #6~�.
}
% ������
% #1 - ������
% #2 - ��������
% #3 - ������
% #4 - ���
% #5 - ��� (�������������)
% #6 - �����
% #7 - ��������
\newcommand{\articleRu}[7]
{
	\item #1 #2 // #3, #4.
	\ifx\\#5\\\else �. #5. \fi
	\ifx\\#6\\\else � #6. \fi
	�.~#7.
}
\newcommand{\articleEn}[7]
{
	\selectlanguage{english}
	\item #1 #2 // #3, #4.
	\ifx\\#5\\\else Vol. #5. \fi
	\ifx\\#6\\\else I. #6. \fi
	P.~#7.
	\selectlanguage{russian}
}
\newcommand{\articleKz}[7]
{
	\item #1 #2 // #3, #4.
	\ifx\\#5\\\else �. #5. \fi
	\ifx\\#6\\\else � #6. \fi
	�.~#7.
}

% �������� ��������
% #1 - ������
% #2 - ��������
% #3 - web-������
\newcommand{\internet}[3]
{
	\item #1 #2. [����������� ������] ����� �������:\\ \texttt{#3}.
}

\newcommand{\refer}[1]
{
	\item #1
}