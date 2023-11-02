\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{msukz}[2016/02/04 Thesis LaTeX class]
\LoadClass[a4paper, 12pt]{extbook}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% НАСТРОЙКИ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage[T2A]{fontenc}		% выходной pdf-текст с корректной кодировкой русского текста
\usepackage[cp1251]{inputenc}		% исходный tex-текст печатается в стандартной WINDOWS кодировке сp-1251
\usepackage[english, russian]{babel}	% переносы текста
%\usepackage{cmap}				% копирование текста из pdf

\usepackage{amssymb}			% amsfonts + дополнительные математические символов
\usepackage{amsthm}				% теоремы  
\usepackage{amsmath}			% многострочные формулы
\usepackage{textcomp}			% большое число типографских значков \No

\usepackage[left=2.5cm, right=2.5cm, top=2cm, bottom=2cm]{geometry} % отступы
\parindent=0.75cm				% отступ параграфа
\binoppenalty=10000				% запретить перенос формул на новую строку по бинарному оператору + - * =
\relpenalty=10000				% запретить перенос формул на новую строку по бинарному отношению > <
\sloppy							% отменить сообщение о переполнении строк, перенося лишние слова на другие строки.

\usepackage{enumitem}
\setlist{leftmargin=\parindent}% выравнить отступы элементов списка

\usepackage{fancyhdr}			
\pagestyle{fancy}				% настройка колонтитулов
\fancyhf{}						% вверху ничего
\fancyfoot[C]{\thepage}			% внизу номер страницы по центру
\renewcommand{\headrulewidth}{0pt} % убрать разделительную линию

\usepackage{listings}			% код программ
\lstset{
	numbers=left, 
	numbersep=5pt, 
    frame=single,
    basicstyle=\ttfamily
}

\theoremstyle{plain}
\newtheorem{theorem}{Теорема}
\newtheorem{lemma}{Лемма}
\newtheorem{corollary}{Следствие}
\newtheorem{definition}{Определение}

\usepackage{graphicx}			% изображения

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ШАПКА ТЕЗИСА
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% #1 - фамилии
% #2 - тема
% #3 - должности
% #4 - организации
% #5 - город, страна
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

% пример
% \thesis
% {Иванов~А.А., Петров~В.В.}
% {Пример оформления тезиса с формулами, рисунками и таблицами}
% {магистрант, студент}
% {Казахстанский филиал МГУ имени М.В.Ломоносова}
% {г.~Астана, Казахстан}
% {ivanov@msu.kz, petrov@msu.kz}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ИЗОБРАЖЕНИЕ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% #1 - имя файла с изображением (без расширения)
% #2 - подпись
\newcommand{\onepicture}[2]
{
	\begin{figure}[h!]
	\begin{center}
		\includegraphics[width=400pt]{#1}
		\caption{#2}
	\end{center}
	\end{figure}
}

% #1 - имя 1-го файла с изображением (без расширения)
% #2 - подпись 1-го изображения
% #1 - имя 2-го файла с изображением (без расширения)
% #2 - подпись 2-го изображения

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
% СПИСОК ЛИТЕРАТУРЫ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newenvironment{references}
{
	\nopagebreak
	~\\
	\textbf{Библиографический список}
	\begin{enumerate}
}{
	\end{enumerate}
}
% КНИГА
% #1 - авторы
% #2 - название
% #3 - город
% #4 - издательство
% #5 - год
% #6 - страниц
\newcommand{\bookRu}[6]
{
	\item #1 #2. #3: #4, #5. #6~с.
}
\newcommand{\bookEn}[6]
{
	\selectlanguage{english}
	\item #1 #2. #3: #4, #5. #6~p.
	\selectlanguage{russian}
}
\newcommand{\bookKz}[6]
{
	\item #1 #2. #3: #4, #5. #6~б.
}
% СТАТЬЯ
% #1 - авторы
% #2 - название
% #3 - журнал
% #4 - год
% #5 - том (необязательно)
% #6 - номер
% #7 - страницы
\newcommand{\articleRu}[7]
{
	\item #1 #2 // #3, #4.
	\ifx\\#5\\\else Т. #5. \fi
	\ifx\\#6\\\else № #6. \fi
	С.~#7.
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
	\ifx\\#5\\\else Т. #5. \fi
	\ifx\\#6\\\else № #6. \fi
	Б.~#7.
}

% ИНТЕРНЕТ источник
% #1 - авторы
% #2 - название
% #3 - web-ссылка
\newcommand{\internet}[3]
{
	\item #1 #2. [Электронный ресурс] режим доступа:\\ \texttt{#3}.
}

\newcommand{\refer}[1]
{
	\item #1
}